import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch as t

from einops import rearrange
from tqdm.auto import tqdm
import fire

from data.activation_capture_prompts.prepare import load_generated_prompts
from data.activation_capture_prompts.definitions import concepts


@dataclass
class SaveActivationsCfg:
    model_name: str = "CompVis/stable-diffusion-v1-4"
    hook_positions: List[str] = field(
        default_factory=lambda: ["unet.up_blocks.1.attentions.1"]
    )
    device: str = "cuda" if t.cuda.is_available() else "cpu"

    prompts_per_batch: int = 30
    max_prompts_per_concept: int = 40
    num_inference_steps: int = 50

    """which concepts to save activations for. defaults to all concepts"""
    concept_indices: List[int] = field(default_factory=lambda: list(range(20)))

    activation_dtype_str: str = "float16"
    save_dir: str = "activations"

    @property
    def activation_dtype(self) -> t.dtype:
        if self.activation_dtype_str == "float16":
            return t.float16
        elif self.activation_dtype_str == "float32":
            return t.float32
        else:
            raise ValueError(
                f"Unsupported activation_dtype_str: {self.activation_dtype_str}. "
                "Supported: 'float16', 'float32'."
            )



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
        target_concepts = set([concepts[i] for i in self.cfg.concept_indices])
        train_prompts = train_prompts[train_prompts["concept"].isin(target_concepts)]

        for c in sorted(target_concepts):
            print(f"starting run for concept {c}")
            prompts = train_prompts[train_prompts["concept"] == c]
            num_prompts = len(prompts)

            if num_prompts == 0:
                print(f"No prompts found for concept {c}, skipping...")
                continue

            if num_prompts > self.cfg.max_prompts_per_concept:
                prompts = prompts.sample(self.cfg.max_prompts_per_concept)
                num_prompts = self.cfg.max_prompts_per_concept

            os.makedirs(self.cfg.save_dir, exist_ok=True)
            save_path = os.path.join(self.cfg.save_dir, f"{c}.bin")
            handle = np.memmap(
                save_path,
                dtype="float16",
                mode="w+",
                shape=(num_prompts * self.cfg.num_inference_steps, 16 * 16, 1280),
            )

            b = self.cfg.prompts_per_batch
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
                start_idx = i * self.cfg.num_inference_steps
                end_idx = min(
                    start_idx + self.cfg.num_inference_steps * b,
                    # if current batch is smaller than batch_size
                    len(handle),
                )
                handle[start_idx:end_idx] = activations.cpu().numpy()

            handle.flush()

    def load_prompts(self) -> pd.DataFrame:
        return load_generated_prompts()


def main(cfg: SaveActivationsCfg = SaveActivationsCfg()):
    runner = SaveActivationsRunner(cfg)
    runner.run()


if __name__ == "__main__":
    fire.Fire(main)
