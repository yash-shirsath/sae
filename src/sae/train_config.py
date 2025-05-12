# adapted from cywinski/saeuron

from dataclasses import dataclass

from simple_parsing import Serializable, list_field

from sae.model import SaeConfig


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig
    dataset_path: list[str] = list_field()

    effective_batch_size: int = 4096
    """Number of activation vectors in a batch."""

    num_workers: int = 4

    persistent_workers: bool = True
    prefetch_factor: int = 2

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = 4e-4
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_scheduler: str = "linear"

    lr_warmup_steps: int = 1000

    auxk_alpha: float = 0.03125
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    feature_sampling_window: int = 100

    hookpoints: list[str] = list_field()
    """List of hookpoints to train SAEs on."""

    distribute_modules: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 5000
    """Save SAEs every `save_every` steps."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1
    wandb_project: str = "sae_stable-diffusion-v1-4"

    def __post_init__(self):
        if self.run_name is None:
            variant = "patch_topk"
            if self.sae.batch_topk:
                variant = "batch_topk"
            elif self.sae.sample_topk:
                variant = "sample_topk"
            self.run_name = f"{variant}_expansion_factor{self.sae.expansion_factor}_k{self.sae.k}_multi_topk{self.sae.multi_topk}_auxk_alpha{self.auxk_alpha}"
