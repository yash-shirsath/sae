import torch as t
from simple_parsing import Serializable, list_field
from transformers import get_scheduler

from sae.dataloader import create_activation_dataloader
from sae.model import Sae, SaeConfig


class TrainConfig(Serializable):
    sae: SaeConfig
    device: str = "cuda"
    dtype: t.dtype = t.float16
    num_epochs: int = 5

    """total activations per batch when patches are flattened"""
    effective_batch_size: int = 4096

    """Base LR. If None, it is automatically chosen based on the number of latents."""
    lr: float | None = 4e-4
    lr_scheduler: str = "linear"
    lr_warmup_steps: int = 0

    activation_dir: str = "activations"


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

        param_groups = {
            "params": self.sae.parameters(),
            "lr": self.cfg.lr or 2e-4 / (self.sae.num_latents / (2**14)) ** 0.5,
            "eps": 6.25e-10,
            "fused": True,
        }

        self.global_step = 0
        self.optimizer = t.optim.Adam(param_groups)
        self.lr_scheduler = get_scheduler(
            name=self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps,
            num_training_steps=(self.num_examples // self.batch_size) * cfg.num_epochs,
        )

    def train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def init_dataloader(self):
        dl = create_activation_dataloader(
            data_dir=self.cfg.activation_dir,
            batch_size=self.cfg.effective_batch_size,
        )
        return dl, len(dl), dl.dataset[0].activation_shape
