"""
Test script for the SAE dataloader.
Generates synthetic data in the same format as the real activation files,
and validates that the loader works correctly.
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sae.dataloader import create_activation_dataloader

# Configuration
TEST_DIR = Path("./test_data")
CONCEPTS = ["Dogs", "Cats", "Cars"]
ACTIVATION_SHAPE = (16 * 16, 1280)  # Same as in the real data
SAMPLES_PER_CONCEPT = {"Dogs": 100, "Cats": 80, "Cars": 60}
DTYPE = "float16"
CLEANUP_TEST_DATA = False  # Set to True to clean up test data after running tests


def generate_test_data():
    """Generate synthetic activation files for testing."""
    os.makedirs(TEST_DIR, exist_ok=True)

    # Check if test data already exists
    all_files_exist = True
    for concept in CONCEPTS:
        if not (TEST_DIR / f"{concept}.bin").exists():
            all_files_exist = False
            break

    if all_files_exist:
        # If all files exist, return concept ranges without regenerating
        concept_sample_map = {}
        for i, concept in enumerate(CONCEPTS):
            base_value = i / 10.0
            concept_sample_map[concept] = (base_value, base_value + 0.1)
        print("Test data already exists, skipping generation.")
        return concept_sample_map

    print("Generating test data...")
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Generate and save fake activation data
    concept_sample_map = {}

    for concept, num_samples in tqdm(
        SAMPLES_PER_CONCEPT.items(),
        desc="Generating concept data",
        total=len(SAMPLES_PER_CONCEPT),
    ):
        # Create unique pattern for each concept so we can verify later
        # For Dogs: values will be in range [0, 0.1]
        # For Cats: values will be in range [0.1, 0.2]
        # For Cars: values will be in range [0.2, 0.3]
        base_value = CONCEPTS.index(concept) / 10.0

        # Create activation data with a recognizable pattern for each concept
        with tqdm(
            total=num_samples, desc=f"Creating {concept} samples", leave=False
        ) as pbar:
            data = np.zeros((num_samples, *ACTIVATION_SHAPE), dtype=DTYPE)

            # Generate data in chunks to show progress
            chunk_size = min(20, num_samples)
            for i in range(0, num_samples, chunk_size):
                end_idx = min(i + chunk_size, num_samples)
                chunk = (
                    np.random.rand(end_idx - i, *ACTIVATION_SHAPE).astype(DTYPE) * 0.1
                    + base_value
                )
                data[i:end_idx] = chunk
                pbar.update(end_idx - i)

        # Save the range of values for each concept (for verification)
        concept_sample_map[concept] = (base_value, base_value + 0.1)

        # Save as memmap file
        memmap_path = TEST_DIR / f"{concept}.bin"
        memmap = np.memmap(
            memmap_path, dtype=DTYPE, mode="w+", shape=(num_samples, *ACTIVATION_SHAPE)
        )

        # Copy the data to the memmap with progress bar
        with tqdm(
            total=num_samples, desc=f"Writing {concept} to disk", leave=False
        ) as pbar:
            chunk_size = min(20, num_samples)
            for i in range(0, num_samples, chunk_size):
                end_idx = min(i + chunk_size, num_samples)
                memmap[i:end_idx] = data[i:end_idx]
                pbar.update(end_idx - i)

        memmap.flush()

    print("Test data generated successfully.")
    return concept_sample_map


def cleanup_test_data():
    """Remove the test data directory."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


def test_dataloader_length_and_indices(concept_ranges):
    """Test that the dataloader has the correct length and indices."""
    # Create a dataloader with equal concept ratios
    dataloader = create_activation_dataloader(
        data_dir=TEST_DIR,
        batch_size=16,
        sample_size=120,  # 40 from each concept
        shuffle=False,  # Disable shuffle for reproducible testing
    )

    # Verify the dataloader has the correct length
    expected_batches = 120 // 16
    if 120 % 16 > 0:
        expected_batches += 1

    assert len(dataloader) == expected_batches, (
        f"Expected {expected_batches} batches, got {len(dataloader)}"
    )

    # Get the dataset
    dataset = dataloader.dataset

    # Verify the dataset has the right number of indices
    assert len(dataset.indices) == 120, (
        f"Expected 120 indices, got {len(dataset.indices)}"
    )

    print("Dataloader length and indices test passed!")
    return True


def test_dataloader_even_concept_distribution(concept_ranges):
    """Test that samples are evenly distributed across concepts."""
    # Create a dataloader with equal concept ratios
    dataloader = create_activation_dataloader(
        data_dir=TEST_DIR,
        batch_size=16,
        sample_size=120,  # 40 from each concept
        shuffle=False,  # Disable shuffle for reproducible testing
    )

    # Verify we have samples from all concepts
    dataset = dataloader.dataset
    concept_counts = {}
    for concept, _ in dataset.indices:
        concept_counts[concept] = concept_counts.get(concept, 0) + 1

    assert len(concept_counts) == len(CONCEPTS), (
        f"Expected samples from {len(CONCEPTS)} concepts, got {len(concept_counts)}"
    )
    assert abs(concept_counts["Dogs"] - 40) <= 1, (
        f"Expected ~40 Dogs samples, got {concept_counts.get('Dogs', 0)}"
    )
    assert abs(concept_counts["Cats"] - 40) <= 1, (
        f"Expected ~40 Cats samples, got {concept_counts.get('Cats', 0)}"
    )
    assert abs(concept_counts["Cars"] - 40) <= 1, (
        f"Expected ~40 Cars samples, got {concept_counts.get('Cars', 0)}"
    )

    print("Concept distribution test passed!")
    return True


def test_dataloader_custom_ratios(concept_ranges):
    """Test the dataloader with custom concept ratios."""

    custom_ratio_loader = create_activation_dataloader(
        data_dir=TEST_DIR,
        batch_size=10,
        concept_ratios={"Dogs": 0.6, "Cats": 0.3, "Cars": 0.1},
        sample_size=100,
        shuffle=False,
    )

    # Get all samples from the dataloader
    all_samples = []
    for batch in custom_ratio_loader:
        all_samples.extend(batch)

    # Count samples by concept based on their value ranges
    concept_counts = {"Dogs": 0, "Cats": 0, "Cars": 0}
    for sample in all_samples:
        sample_np = sample.numpy()
        mean_val = np.mean(sample_np)

        # Determine concept based on value range
        for concept, (min_val, max_val) in concept_ranges.items():
            if min_val <= mean_val < max_val:
                concept_counts[concept] += 1
                break

    # Verify ratios are respected (allowing for small rounding differences)
    assert abs(concept_counts["Dogs"] - 60) <= 1, (
        f"Expected ~60 Dogs samples, got {concept_counts.get('Dogs', 0)}"
    )
    assert abs(concept_counts["Cats"] - 30) <= 1, (
        f"Expected ~30 Cats samples, got {concept_counts.get('Cats', 0)}"
    )
    assert abs(concept_counts["Cars"] - 10) <= 1, (
        f"Expected ~10 Cars samples, got {concept_counts.get('Cars', 0)}"
    )

    print("Custom ratio test passed!")
    return True


def test_dataloader_sample_values(concept_ranges):
    """Test that the loaded samples match their expected concept value ranges."""
    # Create a dataloader with equal concept ratios
    dataloader = create_activation_dataloader(
        data_dir=TEST_DIR,
        batch_size=16,
        sample_size=120,  # 40 from each concept
        shuffle=False,  # Disable shuffle for reproducible testing
    )

    # Get the dataset
    dataset = dataloader.dataset

    # Check several samples directly (not using the dataloader batching)
    num_samples_to_check = min(30, len(dataset.indices))

    for i in range(num_samples_to_check):
        # Get the expected concept directly from the indices list
        expected_concept, _ = dataset.indices[i]

        # Get the actual sample by calling dataset[i]
        sample = dataset[i]
        sample_np = sample.numpy()
        mean_val = np.mean(sample_np)

        # Get the expected range for this concept
        expected_min, expected_max = concept_ranges[expected_concept]

        # Verify the sample's values match the expected range for its claimed concept
        assert expected_min <= mean_val < expected_max, (
            f"Sample {i} claimed to be from '{expected_concept}' but its mean value {mean_val:.4f} "
            f"doesn't match expected range [{expected_min:.4f}, {expected_max:.4f})"
        )

    print("Sample values match their claimed concept sources!")
    return True


def test_dataloader_with_replacement(concept_ranges):
    """Test the dataloader when we request more samples than available."""
    # Create a dataloader requesting more samples than available for Cars
    dataloader = create_activation_dataloader(
        data_dir=TEST_DIR,
        batch_size=16,
        concept_ratios={"Cars": 1.0},  # Only Cars
        sample_size=120,  # More than the 60 Cars samples we have (defined in SAMPLES_PER_CONCEPT)
        shuffle=False,
    )

    # Verify the dataloader has the correct number of samples
    dataset = dataloader.dataset
    assert len(dataset.indices) == 120, (
        f"Expected 120 indices, got {len(dataset.indices)}"
    )

    # Verify all samples are Cars
    concept_counts = {}
    for concept, _ in dataset.indices:
        concept_counts[concept] = concept_counts.get(concept, 0) + 1

    assert list(concept_counts.keys()) == ["Cars"], (
        f"Expected only Cars samples, got {list(concept_counts.keys())}"
    )
    assert concept_counts["Cars"] == 120, (
        f"Expected 120 Cars samples, got {concept_counts.get('Cars', 0)}"
    )

    # Check for duplicates in indices (should have some because of replacement)
    car_indices = [idx for concept, idx in dataset.indices if concept == "Cars"]
    unique_indices = set(car_indices)
    assert len(unique_indices) < 120, (
        f"Expected duplicates in indices, got {len(unique_indices)} unique indices"
    )

    # Verify samples match Cars concept range
    num_samples_to_check = min(30, len(dataset))

    # Get batches from the dataloader instead of accessing dataset directly
    samples_checked = 0
    for batch in dataloader:
        for sample in batch:
            if samples_checked >= num_samples_to_check:
                break

            # Convert tensor to numpy for analysis
            sample_np = sample.numpy()
            mean_val = np.mean(sample_np)

            # Get the expected range for Cars
            expected_min, expected_max = concept_ranges["Cars"]

            # Verify the sample's values match the expected range for Cars
            assert expected_min <= mean_val < expected_max, (
                f"Sample {samples_checked} should be from 'Cars' but its mean value {mean_val:.4f} "
                f"doesn't match expected range [{expected_min:.4f}, {expected_max:.4f})"
            )

            samples_checked += 1

        if samples_checked >= num_samples_to_check:
            break

    print("Replacement sampling test passed!")
    return True


if __name__ == "__main__":
    # Generate test data once
    try:
        concept_ranges = generate_test_data()
        print(concept_ranges)

        # Run all tests with the same test data
        print("\nRunning dataloader length and indices test...")
        test_dataloader_length_and_indices(concept_ranges)

        print("\nRunning concept distribution test...")
        test_dataloader_even_concept_distribution(concept_ranges)

        print("\nRunning custom ratio test...")
        test_dataloader_custom_ratios(concept_ranges)

        print("\nRunning sample values test...")
        test_dataloader_sample_values(concept_ranges)

        print("\nRunning replacement sampling test...")
        test_dataloader_with_replacement(concept_ranges)

        print("\nAll tests completed successfully!")
    finally:
        # Clean up test data at the end if flag is set
        if CLEANUP_TEST_DATA:
            print("Cleaning up test data...")
            cleanup_test_data()
        else:
            print("Test data preserved (CLEANUP_TEST_DATA=False)")
