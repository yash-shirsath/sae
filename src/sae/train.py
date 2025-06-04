from collections import defaultdict
from dataclasses import asdict
from time import time

import psutil
import torch as t
from matplotlib import pyplot as plt
from safetensors.torch import load_model
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from transformers import get_scheduler  # type: ignore

from dataloader import create_activation_dataloader
from sae.model import Sae
from sae.train_config import TrainConfig


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.dataloader, self.num_examples, (self.patch_size, self.d_in) = (
            self.init_dataloader()
        )
        self.device = t.device(self.cfg.device)

        self.sae = Sae(
            cfg=self.cfg.sae,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        self.effective_batch_size = self.cfg.effective_batch_size
        self.batch_size = self.effective_batch_size // self.patch_size
        print(f"Effective batch size: {self.effective_batch_size}")
        print(f"Batch size: {self.batch_size}")
        print(self.sae)

        self.increment_tokens = self.effective_batch_size

        param_groups = {
            "params": self.sae.parameters(),
            "lr": self.cfg.lr or 2e-4 / (self.sae.num_latents / (2**14)) ** 0.5,
            "eps": 6.25e-10,
            "fused": True,
        }

        self.global_step = 0
        self.optimizer = t.optim.Adam(param_groups["params"], **param_groups)
        self.lr_scheduler = get_scheduler(
            name=self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps,
            num_training_steps=(self.num_examples // self.batch_size) * cfg.num_epochs,
        )

    def train(self):
        # Use Tensor Cores even for fp32 matmuls
        t.set_float32_matmul_precision("high")

        if self.cfg.log_to_wandb:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project=self.cfg.wandb_project,
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(p.numel() for p in self.sae.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")

        num_batches = (self.num_examples // self.batch_size) * self.cfg.num_epochs

        device = t.device(self.cfg.device)

        pbar = tqdm(
            desc="Training",
            initial=self.global_step,
            total=num_batches,
        )

        did_fire = {
            name: t.zeros(self.sae.num_latents, device=device, dtype=t.bool)
            for name in self.cfg.hookpoints
        }
        num_tokens_in_step = 0
        total_tokens = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_l0 = defaultdict(float)
        avg_l2 = defaultdict(float)
        avg_exp_var_mean = defaultdict(float)
        avg_exp_var_std = defaultdict(float)

        frac_active_list = []  # track active features

        # Initialize disk I/O stats
        self.initial_disk_io = psutil.disk_io_counters()

        for _ in range(self.cfg.num_epochs):
            for batch in self.dataloader:
                hidden_dict = {}
                start_loading = time()
                for hook in self.cfg.hookpoints:
                    hidden_dict[hook] = batch
                data_loading_time = time() - start_loading

                # Bookkeeping for dead feature detection
                num_tokens_in_step += self.increment_tokens
                total_tokens += self.increment_tokens

                for name, hiddens in hidden_dict.items():
                    # On the first iteration, initialize the decoder bias
                    if self.global_step == 0:
                        # NOTE: The all-cat here could conceivably cause an OOM in some
                        # cases, but it's unlikely to be a problem with small world sizes.
                        # We could avoid this by "approximating" the geometric median
                        # across all ranks with the mean (median?) of the geometric medians
                        # on each rank. Not clear if that would hurt performance.
                        hiddens_input = hiddens.view(-1, hiddens.shape[-1])
                        median = geometric_median(hiddens_input)
                        median = median.to(self.sae.device)
                        self.sae.b_dec.data = median.to(self.sae.dtype)

                    # Make sure the W_dec is still unit-norm
                    if self.sae.cfg.normalize_decoder:
                        self.sae.set_decoder_norm_to_unit_norm()

                    acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                    denom = acc_steps * self.cfg.wandb_log_frequency

                    # Save memory by chunking the activations
                    for chunk_hiddens in hiddens.chunk(self.cfg.micro_acc_steps):
                        chunk_hiddens = chunk_hiddens.to(device)
                        out = self.sae(
                            chunk_hiddens,
                            dead_mask=(
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                                if self.cfg.auxk_alpha > 0
                                else None
                            ),
                        )

                        avg_fvu[name] += float(out.fvu.detach() / denom)
                        avg_l0[name] += float(out.l0_loss.detach() / denom)
                        avg_l2[name] += float(out.l2_loss.detach() / denom)
                        avg_exp_var_mean[name] += float(
                            out.explained_variance.mean().item() / denom
                        )
                        avg_exp_var_std[name] += float(
                            out.explained_variance.std().item() / denom
                        )
                        if self.cfg.auxk_alpha > 0:
                            avg_auxk_loss[name] += float(out.auxk_loss.detach() / denom)

                        loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
                        loss.div(acc_steps).backward()

                        # Update the did_fire mask
                        did_fire[name][out.latent_indices.flatten()] = True

                    clip_grad_norm_(self.sae.parameters(), 1.0)

                # Check if we need to actually do a training step
                step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)

                if self.cfg.sae.normalize_decoder:
                    self.sae.remove_gradient_parallel_to_decoder_directions()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                    with t.no_grad():
                        # Update the dead feature mask
                        for name, counts in self.num_tokens_since_fired.items():
                            counts += num_tokens_in_step
                            counts[did_fire[name]] = 0

                    if (
                        self.cfg.log_to_wandb
                        and (step + 1) % self.cfg.wandb_log_frequency == 0
                    ):
                        info = {}
                        did_fire_counts = {
                            name: mask.sum().item() for name, mask in did_fire.items()
                        }
                        did_fire_percentages = {
                            name: count / self.sae.num_latents
                            for name, count in did_fire_counts.items()
                        }

                        for name in self.cfg.hookpoints:
                            mask = (
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                            )
                            fire_count = t.zeros(self.sae.num_latents, dtype=t.long)
                            unique, unique_counts = t.unique(
                                out.latent_indices.flatten(),
                                return_counts=True,
                            )
                            fire_count[unique] = unique_counts.cpu()
                            frac_active_list.append(fire_count)

                            if len(frac_active_list) > self.cfg.feature_sampling_window:
                                frac_active_in_window = t.stack(
                                    frac_active_list[
                                        -self.cfg.feature_sampling_window :
                                    ],
                                    dim=0,
                                )
                                feature_sparsity = frac_active_in_window.sum(0) / (
                                    self.cfg.feature_sampling_window
                                    * self.effective_batch_size
                                )
                            else:
                                frac_active_in_window = t.stack(frac_active_list, dim=0)
                                feature_sparsity = frac_active_in_window.sum(0) / (
                                    len(frac_active_list) * self.effective_batch_size
                                )

                            log_feature_sparsity = t.log10(feature_sparsity + 1e-8)

                            info.update(
                                {
                                    f"fvu/{name}": avg_fvu[name],
                                    f"l0/{name}": avg_l0[name],
                                    f"l2/{name}": avg_l2[name],
                                    f"explained_variance/{name}": avg_exp_var_mean[
                                        name
                                    ],
                                    f"explained_variance_std/{name}": avg_exp_var_std[
                                        name
                                    ],
                                    f"dead_pct/{name}": mask.mean(
                                        dtype=t.float32
                                    ).item(),
                                    f"lr/{name}": self.optimizer.param_groups[0]["lr"],
                                    f"fire_pct/{name}": did_fire_percentages[name],
                                    f"sparsity_below_1e-2/{name}": (
                                        feature_sparsity < 1e-2
                                    )
                                    .float()
                                    .mean()
                                    .item(),
                                    f"sparsity_below_1e-3/{name}": (
                                        feature_sparsity < 1e-3
                                    )
                                    .float()
                                    .mean()
                                    .item(),
                                    f"sparsity_below_1e-4/{name}": (
                                        feature_sparsity < 1e-4
                                    )
                                    .float()
                                    .mean()
                                    .item(),
                                    f"sparsity_below_1e-5/{name}": (
                                        feature_sparsity < 1e-5
                                    )
                                    .float()
                                    .mean()
                                    .item(),
                                    "total_tokens": total_tokens,
                                    "data_load_time": data_loading_time,
                                }
                            )
                            if self.cfg.auxk_alpha > 0:
                                info[f"auxk/{name}"] = avg_auxk_loss[name]

                            if (step + 1) % (self.cfg.wandb_log_frequency * 10) == 0:
                                plt.hist(
                                    log_feature_sparsity.tolist(),
                                    bins=50,
                                    color="blue",
                                    alpha=0.7,
                                )
                                plt.title("Feature Density")
                                plt.xlabel("Log Feature Density")
                                plt.tight_layout()
                                info[f"feature_density/{name}"] = wandb.Image(plt.gcf())
                                plt.close()

                        avg_auxk_loss.clear()
                        avg_fvu.clear()
                        avg_l0.clear()
                        avg_l2.clear()
                        avg_exp_var_mean.clear()
                        avg_exp_var_std.clear()

                        wandb.log(info, step=step)

                    # Reset stats for this step
                    with t.no_grad():
                        num_tokens_in_step = 0
                        for mask in did_fire.values():
                            mask.zero_()

                    if (
                        self.cfg.save_every > 0
                        and (step + 1) % self.cfg.save_every == 0
                    ):
                        self.save()

                self.global_step += 1
                pbar.update()
        self.save()
        pbar.close()

    def save(self):
        """Save the SAEs to disk."""

        path = (
            f"sae-ckpts/{self.cfg.wandb_project}/{self.cfg.run_name}"
            if self.cfg.run_name
            else f"sae-ckpts/{self.cfg.wandb_project}"
        )

        print("Saving checkpoint")

        for hook in self.cfg.hookpoints:
            self.sae.save_to_disk(f"{path}/{hook}")

        t.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
        t.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
        t.save(
            {
                "global_step": self.global_step,
                "num_tokens_since_fired": self.num_tokens_since_fired,
            },
            f"{path}/state.pt",
        )

        self.cfg.save_json(f"{path}/config.json")

    def load(self, path: str):
        """Load the trainer state from disk."""
        device = self.cfg.device

        # Load the train state first so we can print the step number
        train_state = t.load(f"{path}/state.pt", map_location=device, weights_only=True)
        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = train_state["num_tokens_since_fired"]

        print(
            f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m"
        )

        lr_state = t.load(
            f"{path}/lr_scheduler.pt", map_location=device, weights_only=True
        )
        opt_state = t.load(
            f"{path}/optimizer.pt", map_location=device, weights_only=True
        )
        self.optimizer.load_state_dict(opt_state)
        self.lr_scheduler.load_state_dict(lr_state)

        load_model(self.sae, f"{path}/sae.safetensors", device=str(device))

    def init_dataloader(self):
        dl = create_activation_dataloader(
            data_dir=self.cfg.activation_dir,
            batch_size=self.cfg.effective_batch_size,
        )
        return dl, len(dl), dl.dataset[0].activation_shape


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            print("Early stopping in computation of the geometric median")
            break

    return guess
