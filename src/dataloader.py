import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ActivationDataset(Dataset):
    """Dataset for loading activations from memmap files with tunable concept ratios."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        concept_ratios: Optional[Dict[str, float]] = None,
        sample_size: Optional[int] = None,
        activation_shape: Tuple[int, int] = (16 * 16, 1280),
        dtype: str = "float16",
        seed: int = 42,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the memmap files
            concept_ratios: Dict mapping concept names to sampling ratios (will be normalized to sum to 1)
            sample_size: Total number of samples to use for training. If None, uses all available samples.
            activation_shape: Shape of each activation
            dtype: Data type of the activations
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.activation_shape = activation_shape
        self.dtype = dtype
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Find all memmap files in the data directory
        memmap_files = list(self.data_dir.glob("*.bin"))
        self.concepts = [f.stem for f in memmap_files]

        if not self.concepts:
            raise ValueError(f"No memmap files found in {data_dir}")

        # Setup concept ratios
        if concept_ratios is None:
            # Equal distribution by default
            self.concept_ratios = {
                concept: 1.0 / len(self.concepts) for concept in self.concepts
            }
        else:
            # Filter out concepts that don't have memmap files
            valid_concepts = {
                k: v for k, v in concept_ratios.items() if k in self.concepts
            }
            if not valid_concepts:
                raise ValueError(
                    f"None of the provided concepts {list(concept_ratios.keys())} have memmap files"
                )

            # Normalize ratios to sum to 1
            total = sum(valid_concepts.values())
            self.concept_ratios = {k: v / total for k, v in valid_concepts.items()}

        # Load memmap files
        self.memmaps = {}
        self.concept_sizes = {}

        for concept in self.concepts:
            memmap_path = self.data_dir / f"{concept}.bin"
            if not memmap_path.exists():
                continue

            memmap = np.memmap(
                memmap_path,
                dtype=self.dtype,
                mode="r",
                shape=(self._get_memmap_size(memmap_path),) + self.activation_shape,
            )

            self.memmaps[concept] = memmap
            self.concept_sizes[concept] = len(memmap)

        if sample_size is None:
            self.total_sample_size = sum(self.concept_sizes.values())
        else:
            self.total_sample_size = sample_size

        self.samples_per_concept = {
            concept: int(ratio * self.total_sample_size)
            for concept, ratio in self.concept_ratios.items()
        }

        # Adjust to ensure we get exactly total_sample_size samples
        leftover = self.total_sample_size - sum(self.samples_per_concept.values())
        if leftover > 0:
            # Add the leftover samples to concepts in order of their ratios
            sorted_concepts = sorted(
                self.concept_ratios.keys(),
                key=lambda c: self.concept_ratios[c],
                reverse=True,
            )
            for i in range(leftover):
                self.samples_per_concept[sorted_concepts[i % len(sorted_concepts)]] += 1

        # Generate indices for each concept
        self.indices = []
        for concept, n_samples in self.samples_per_concept.items():
            if concept not in self.memmaps:
                continue

            concept_size = self.concept_sizes[concept]
            if n_samples > concept_size:
                # If we need more samples than available, sample with replacement
                concept_indices = np.random.choice(
                    concept_size, size=n_samples, replace=True
                )
            else:
                # Otherwise, sample without replacement
                concept_indices = np.random.choice(
                    concept_size, size=n_samples, replace=False
                )

            self.indices.extend([(concept, idx) for idx in concept_indices])

    def _get_memmap_size(self, memmap_path: Path) -> int:
        """Get the number of samples in a memmap file."""
        file_size = memmap_path.stat().st_size
        element_size = np.dtype(self.dtype).itemsize
        total_elements_per_sample = np.prod(self.activation_shape)
        return file_size // (element_size * total_elements_per_sample)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        concept, sample_idx = self.indices[idx]
        activation = self.memmaps[concept][sample_idx]

        # Convert to torch tensor and handle dtype
        activation = torch.from_numpy(activation)
        return activation


def create_activation_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    concept_ratios: Optional[Dict[str, float]] = None,
    sample_size: Optional[int] = None,
    activation_shape: Tuple[int, int] = (16 * 16, 1280),
    dtype: str = "float16",
    seed: int = 42,
    num_workers: int = 8,
    prefetch_factor: int = 10,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader[ActivationDataset]:
    """
    Create a dataloader for SAE training with tunable concept ratios.

    Args:
        data_dir: Directory containing the memmap files
        batch_size: Batch size for training
        concept_ratios: Dict mapping concept names to sampling ratios
        sample_size: Total number of samples to use for training. If None, uses all available samples.
        activation_shape: Shape of each activation
        dtype: Data type of the activations
        seed: Random seed for reproducibility
        num_workers: Number of workers for the dataloader
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader for SAE training
    """
    dataset = ActivationDataset(
        data_dir=data_dir,
        concept_ratios=concept_ratios,
        sample_size=sample_size,
        activation_shape=activation_shape,
        dtype=dtype,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


if __name__ == "__main__":
    # Example usage
    dataloader = create_activation_dataloader(
        data_dir="activations",
        batch_size=50,
        concept_ratios={"Dogs": 0.7, "Cats": 0.3},
    )

    print(f"Total batches: {len(dataloader)}")

    # Prefetch first batch
    dataloader_iter = iter(dataloader)
    next_batch = next(dataloader_iter)
    device = "cuda"
    next_batch = next_batch.to(device, non_blocking=True)

    for i in range(len(dataloader)):
        # Current batch is the previously prefetched batch
        batch = next_batch

        # Prefetch next batch if not on the last iteration
        try:
            next_batch = next(dataloader_iter)
            next_batch = next_batch.to(device, non_blocking=True)
        except StopIteration:
            break

        print(f"Batch {i} shape: {batch.shape}")
