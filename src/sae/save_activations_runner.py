import os
import random
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as t
from datasets import (
    Array2D,
    Array4D,
    Dataset,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
from diffusers import DDIMScheduler, StableDiffusionPipeline  # type: ignore
from einops import rearrange
from tqdm.auto import tqdm

from data.activation_capture_prompts.prepare import (
    balance_concepts_styles,
    load_generated_prompts,
)


@dataclass
class SaveActivationsCfg:
    model_name: str = "CompVis/stable-diffusion-v1-4"
    hook_positions: List[str] = field(
        default_factory=lambda: ["unet.up_blocks.1.attentions.1"]
    )
    device: str = "cuda"

    batch_size: int = 30
    subset_size: int = 5_000  # bigger than the biggest prompt len for now
    activation_dtype = t.float16

    """biased dataset toward main_object"""
    main_object = "Dog"

    """whether this instance of the runner will process only dogs"""
    only_dog = True


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
            self._register_cache_hook(position, position_activation_map)
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
            assert (s, c, h, w) == (50, 1280, 16, 16), (
                f"unexpected activation shape: {activations.shape}"
            )
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

    def _register_cache_hook(self, position: str, cache_output: dict):
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
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_name,
            torch_dtype=cfg.activation_dtype,
            safety_checker=None,
        )
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

        train_prompts = self.load_prompts()

        if self.cfg.only_dog:
            target_concepts = {"Dogs"}
        else:
            target_concepts = set(train_prompts["concept"].unique()) - {"Dogs"}

        for c in target_concepts:
            print(f"starting run for concept {c}")
            prompts = train_prompts[train_prompts["concept"] == c]
            num_prompts = len(prompts)

            if num_prompts > self.cfg.subset_size:
                prompts = prompts.sample(self.cfg.subset_size)
                num_prompts = self.cfg.subset_size

            handle = np.memmap(
                f"{c}.bin",
                dtype="float16",
                mode="w+",
                shape=(num_prompts, 1280 * 50, 16, 16),
            )

            b = self.cfg.batch_size
            for i in tqdm(
                range(0, num_prompts, b),
                desc=f"Processing {c} prompts",
                total=num_prompts // b + (1 if num_prompts % b else 0),
            ):
                batch_df = prompts[i : i + b]
                activations = pipe.run_with_cache(
                    prompt=batch_df["prompt"].tolist(),
                    positions_to_cache=self.cfg.hook_positions,
                )
                activations = activations["unet.up_blocks.1.attentions.1"]
                handle[i : i + b] = activations.cpu().numpy()

            handle.flush()

    def load_prompts(self) -> pd.DataFrame:
        all = load_generated_prompts()
        balanced = balance_concepts_styles(all, main_concept="Dogs", random_state=42)
        return balanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-dog", action="store_true", help="Only process dog-related prompts"
    )
    args = parser.parse_args()

    cfg = SaveActivationsCfg()
    cfg.only_dog = args.only_dog

    if not cfg.only_dog:
        cfg.device = "cuda:1"

    runner = SaveActivationsRunner(cfg)
    runner.run()
