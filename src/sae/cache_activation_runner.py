from collections import defaultdict
from typing import Union, List, Optional, Tuple
from diffusers import StableDiffusionPipeline  # type: ignore
from dataclasses import dataclass
import torch as t


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
        pipe.to(self.cfg.device)

        pipe.run_with_cache("Sick image of clouds")


class HookedDiffusionPipeline:
    def __init__(self, pipe: StableDiffusionPipeline) -> None:
        self.pipe = pipe

    def run_with_cache(
        self,
        prompt: Union[str, List[str]] = "",
        positions_to_cache: List[str] = [],
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[Union[t.Generator, List[t.Generator]]] = None,
    ) -> Tuple[dict, dict]:
        cache_input, cache_output = defaultdict(list), defaultdict(list)
        positions_to_cache = ["unet.up_blocks.1.attentions.1"]

        hook_handles = [
            self._register_cache(position, cache_input, cache_output)
            for position in positions_to_cache
        ]
        hook_handles = [h for h in hook_handles if h]

        self._run_generation_pipeline(
            prompt, guidance_scale, num_inference_steps, generator
        )

        for handle in hook_handles:
            handle.remove()

        return cache_input, cache_output

    def _run_generation_pipeline(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: float,
        num_inference_steps: int,
        generator: Optional[Union[t.Generator, List[t.Generator]]],
    ):
        self.pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    def _register_cache(self, position: str, cache_input: dict, cache_output: dict):
        def hook_fn(module, input, output):
            cache_input[position].append(input)
            cache_output[position].append(output)

        block: t.nn.Module = self._locate_block(position)
        return block.register_forward_hook(hook_fn)

    def _locate_block(self, position: str):
        block = self.pipe
        for step in position.split("."):
            if step.isdigit():
                step = int(step)
                block = block[step]
            else:
                block = getattr(block, step)
        return block

    def to(self, device: str):
        self.pipe.to(device)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "HookedDiffusionPipeline":
        pipe = StableDiffusionPipeline.from_pretrained(model_name)
        assert isinstance(pipe, StableDiffusionPipeline)
        return cls(pipe)


if __name__ == "__main__":
    cfg = CacheActivationsRunnerCfg()
    runner = CacheActivationsRunner(cfg)
    runner.run()
