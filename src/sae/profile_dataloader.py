# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import time
import psutil
import torch
from torch.utils.data import DataLoader
from sae.dataloader import create_activation_dataloader
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np


# %%


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB


def profile_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_batches: int = 100,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Profile the dataloader performance.

    Args:
        data_dir: Directory containing activation files
        batch_size: Batch size for the dataloader
        num_batches: Number of batches to profile
        num_workers: Number of worker processes

    Returns:
        Tuple of (batch_times, memory_usage, throughput)
    """
    # Initialize lists to store metrics
    batch_times = []
    memory_usage = []
    throughput = []

    # Create dataloader
    dataloader = create_activation_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        # concept_ratios={"Cats": 1},
    )

    # Warm up
    print("Warming up...")
    for _ in range(5):
        next(iter(dataloader))

    # Profile
    print(f"Profiling {num_batches} batches...")
    start_time = time.time()
    total_samples = 0

    batch_start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_size = batch.shape[0]
        total_samples += batch_size

        # Record metrics
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        memory_usage.append(get_memory_usage())
        throughput.append(batch_size / (batch_time + 0.01))

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} batches")
        batch_start = time.time()

    total_time = time.time() - start_time
    print(f"\nProfiling complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average batch time: {np.mean(batch_times):.4f}s")
    print(f"Average throughput: {np.mean(throughput):.2f} samples/s")
    print(f"Peak memory usage: {max(memory_usage):.2f} GB")

    return batch_times, memory_usage, throughput


def plot_metrics(
    batch_times: List[float],
    memory_usage: List[float],
    throughput: List[float],
    save_path: str = "dataloader_profile.png",
):
    """Plot the profiling metrics."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot batch times
    ax1.plot(batch_times)
    ax1.set_title("Batch Loading Time")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Time (s)")

    # Plot memory usage
    ax2.plot(memory_usage)
    ax2.set_title("Memory Usage")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Memory (GB)")

    # Plot throughput
    ax3.plot(throughput)
    ax3.set_title("Throughput")
    ax3.set_xlabel("Batch")
    ax3.set_ylabel("Samples/s")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Profile plot saved to {save_path}")


# %%
if __name__ == "__main__":
    # Get absolute path to activations directory
    activations_dir = Path(__file__).parent.parent.parent / "activations"

    # Profile the dataloader
    batch_times, memory_usage, throughput = profile_dataloader(
        data_dir=str(activations_dir),
        batch_size=32,
        num_batches=100,
        num_workers=8,
        prefetch_factor=10,
    )

    # Plot the results
    plot_metrics(batch_times, memory_usage, throughput)

# %%
