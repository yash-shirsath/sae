from typing import Union, List, Optional
from diffusers import StableDiffusionPipeline
from dataclasses import dataclass
import torch


@dataclass
class CacheActivationsRunnerCfg:
    model_name: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda"


class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerCfg) -> None:
        self.cfg = cfg

    def run(self):
        """
        takes pipeline and dataset, runs stable diffusion
        """
        pipe = HookedDiffusionPipeline.from_pretrained(self.cfg.model_name)


class HookedDiffusionPipeline:
    def __init__(self, pipe: StableDiffusionPipeline) -> None:
        self.pipe = pipe

    def run_with_cache(
        self,
        prompt: Union[str, List[str]] = "",
        positions_to_cache: List[str] = [],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        positions_to_cache = ["unet.up_blocks.1.attentions.1"]

    @classmethod
    def from_pretrained(cls, model_name: str) -> "HookedDiffusionPipeline":
        pipe = StableDiffusionPipeline.from_pretrained(model_name)
        return cls(pipe)
