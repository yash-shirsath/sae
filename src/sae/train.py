from sae.dataloader import create_activation_dataloader
from sae.model import Sae, SaeConfig
from simple_parsing import Serializable, list_field
import torch as t


class TrainConfig(Serializable):
    sae: SaeConfig
    device: str = "cuda"
    dtype: t.dtype = t.float16
    """total activations per batch when patches are flattened"""
    effective_batch_size: int = 4096

    activation_dir: str = "activations"


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.dataloader = self.init_dataloader()
        self.device = t.device(self.cfg.device)

        self.sae = Sae(
            cfg=self.cfg.sae,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def init_dataloader(self):
        return create_activation_dataloader(
            data_dir=self.cfg.activation_dir,
            batch_size=self.cfg.effective_batch_size,
        )
