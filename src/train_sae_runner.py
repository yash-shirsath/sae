import pprint
from dataclasses import asdict

import fire
import torch as t

from sae.model import SaeConfig
from sae.train import Trainer
from sae.train_config import TrainConfig


def main(**kwargs):
    """
    Main function to train a Sparse Autoencoder.

    Example usage:
    python src/train_sae_runner.py --activation_dir="path/to/activations" --d_in=768 --expansion_factor=4 ...
    """
    # Separate kwargs for SaeConfig and TrainConfig
    sae_config_kwargs = {k: v for k, v in kwargs.items() if hasattr(SaeConfig, k)}
    train_config_kwargs = {
        k: v for k, v in kwargs.items() if hasattr(TrainConfig, k) and k != "sae"
    }

    # Create SaeConfig and TrainConfig instances
    sae_config = SaeConfig(**sae_config_kwargs)

    if "dtype" in train_config_kwargs:
        dtype_str = train_config_kwargs["dtype"]
        if dtype_str == "float16":
            train_config_kwargs["dtype"] = t.float16
        elif dtype_str == "float32":
            train_config_kwargs["dtype"] = t.float32
        else:
            print(
                f"Warning: dtype '{dtype_str}' not recognized or not a torch.dtype. Defaulting to float32. Valid options: float16, float32, bfloat16"
            )
            train_config_kwargs["dtype"] = t.float32

    train_config = TrainConfig(sae=sae_config, **train_config_kwargs)

    print("--- TrainConfig ---")
    config_dict = asdict(train_config)
    pprint.pprint(config_dict, indent=2)
    print("--------------------")

    trainer = Trainer(cfg=train_config)
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
