import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import torch as t
from datasets import (
    Array2D,
    Array4D,
    Features,
    IterableDataset,
    Sequence,
    Value,
    load_from_disk,
)
from diffusers import DDIMScheduler, StableDiffusionPipeline  # type: ignore
from einops import rearrange


@dataclass
class SaveActivationsCfg:
    model_name: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda"

    batch_size: int = 32
    activation_dtype = t.float16


class HookedDiffusionPipeline:
    def __init__(self, pipe: StableDiffusionPipeline, cfg: SaveActivationsCfg) -> None:
        self.pipe = pipe
        self.cfg = cfg

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

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
            # stacks list of tensors into a step dimension
            activations = t.stack(position_activation_map[position], dim=0)
            s, b, c, h, w = activations.shape
            assert (s, c, h, w) == (51, 1280, 16, 16)
            activations = rearrange(activations, "steps b c h w -> b (steps c) h w")
            cache_output[position] = activations

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
            activations = output[0].to(self.cfg.activation_dtype)
            _, c, h, w = activations.shape
            assert (c, h, w) == (1280, 16, 16), "unexpected activation shape"
            neg_conditioned, text_conditioned = activations.chunk(2, dim=0)
            cache_output[position].append(text_conditioned)

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
    def from_pretrained(cls, cfg: SaveActivationsCfg) -> "HookedDiffusionPipeline":
        pipe = StableDiffusionPipeline.from_pretrained(cfg.model_name)
        assert isinstance(pipe, StableDiffusionPipeline)
        return cls(pipe, cfg)


class SaveActivationsRunner:
    def __init__(self, cfg: SaveActivationsCfg) -> None:
        self.cfg = cfg

    def run(self):
        """
        takes pipeline and dataset, runs stable diffusion
        """
        pipe = HookedDiffusionPipeline.from_pretrained(self.cfg)
        pipe.to(self.cfg.device)

        activations = pipe.run_with_cache(
            prompt=[
                "Sick image of clouds",
                "Sick image of clouds",
                "Sick image of clouds",
            ],
            positions_to_cache=[
                "unet.up_blocks.1.attentions.1",
            ],
        )
        print(activations["unet.up_blocks.1.attentions.1"].shape)
        # prompts_dict = self.load_prompts()
        # train_prompts = self.subset_prompts(prompts_dict)

        # ds = IterableDataset.from_generator(
        #     self.activation_generator(pipe, train_prompts),
        #     features=Features(
        #         {
        #             "activations": Array4D(
        #                 shape=(2, 1280 * 51, 16, 16), dtype="float16"
        #             ),
        #         }
        #     ),
        # )

    # def subset_prompts(self, prompt_dict: dict) -> list[str]:
    #     pass

    # def load_prompts(self, data_dir="") -> dict:
    #     pass

    # def activation_generator(
    #     self, pipe: HookedDiffusionPipeline, prompts: list[str]
    # ) -> Iterable[dict]:
    #     """
    #     1. generates batches
    #     2. yields activations
    #     """
    #     b = self.cfg.batch_size
    #     for i in range(0, len(prompts), b):
    #         batch_prompts = prompts[i : i + b]
    #         activations = pipe.run_with_cache(
    #             prompt=batch_prompts,
    #             positions_to_cache=[
    #                 "unet.up_blocks.1.attentions.1",
    #                 "unet.up_blocks.1.attentions.2",
    #             ],
    #         )
    #     yield {"activations": activations}

    # def save_activations(
    #     self, activations: dict, output_dir=None, file_prefix="activations"
    # ):
    #     dir = Path(__file__).parent
    #     if output_dir is None:
    #         dir = os.path.join(dir, "activations")

    #     os.makedirs(dir, exist_ok=True)

    #     for position in activations.keys():
    #         out = activations[position].cpu()
    #         output_path = os.path.join(dir, f"{file_prefix}_{position}.pt")
    #         t.save(out, output_path)

    #     print(f"Saved {len(activations)} activations to {dir}")
    #     return True


if __name__ == "__main__":
    cfg = SaveActivationsCfg()
    runner = SaveActivationsRunner(cfg)
    runner.run()
