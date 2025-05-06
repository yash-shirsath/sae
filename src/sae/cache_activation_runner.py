from collections import defaultdict
from typing import Union, List, Optional, Tuple
from diffusers import StableDiffusionPipeline  # type: ignore
from dataclasses import dataclass
import torch as t
import os
from pathlib import Path


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

        output_activations = pipe.run_with_cache(
            prompt="Sick image of clouds",
            positions_to_cache=[
                "unet.up_blocks.1.attentions.1",
                "unet.up_blocks.1.attentions.2",
            ],
        )
        self.save_activations(output_activations)

    def save_activations(
        self, activations: dict, output_dir=None, file_prefix="activations"
    ):
        dir = Path(__file__).parent
        if output_dir is None:
            dir = os.path.join(dir, "activations")

        os.makedirs(dir, exist_ok=True)

        for position in activations.keys():
            out = activations[position].cpu()
            output_path = os.path.join(dir, f"{file_prefix}_{position}.pt")
            t.save(out, output_path)

        print(f"Saved {len(activations)} activations to {dir}")
        return True


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
    ) -> dict:
        position_activation_map = defaultdict(list)

        hook_handles = [
            self._register_cache(position, position_activation_map)
            for position in positions_to_cache
        ]
        hook_handles = [h for h in hook_handles if h]

        self._run_generation_pipeline(
            prompt, guidance_scale, num_inference_steps, generator
        )

        for handle in hook_handles:
            handle.remove()

        cache_output = {}
        # Stack all tensors after hooks are removed
        for position in position_activation_map:
            cache_output[position] = t.stack(position_activation_map[position], dim=0)
            assert cache_output[position].shape == (51, 2, 1280, 16, 16)

        return cache_output

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

    def _register_cache(self, position: str, cache_output: dict):
        def hook_fn(module, input, output):
            assert isinstance(output, tuple) and len(output) == 1, (
                "unexpected output from hook"
            )
            cache_output[position].append(output[0])

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
