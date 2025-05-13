from sae.model import Sae
from simple_parsing import Serializable, list_field
import torch as t
from sae.model import SaeConfig


class TrainConfig(Serializable):
    sae: SaeConfig

    """total activations per batch when patches are flattened"""
    effective_batch_size: int = 4096

    activation_dir: str = "activations"


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.dataloader = self.init_dataloader()

    def train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def init_dataloader(self):
        return create_activation_dataloader(
            data_dir="activations",
            batch_size=50,
            concept_ratios={"Dogs": 0.7, "Cats": 0.3},
        )
