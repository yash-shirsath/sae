# %%
from typing import Optional

import pandas as pd


def subset_prompts(
    df: pd.DataFrame, main_object: str, random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Subset the prompts DataFrame to create a balanced dataset with a main object and other objects.
    Ensures even distribution of both objects and styles in the non-main object portion.

    Args:
        df: Input DataFrame containing prompts with 'object' and 'style' columns
        main_object: The main object to include all prompts for
        random_state: Optional random seed for reproducibility

    Returns:
        DataFrame containing:
        - All prompts for the main object
        - An even distribution of other objects and their styles
    """
    # Get all prompts for the main object
    main_object_df = df[df["object"] == main_object].copy()
    main_object_count = len(main_object_df)

    # Get all other objects
    other_objects_df = df[df["object"] != main_object].copy()

    # Get unique objects and styles
    unique_objects = other_objects_df["object"].unique()
    unique_styles = other_objects_df["style"].unique()
    n_objects = len(unique_objects)
    n_styles = len(unique_styles)

    # Calculate how many prompts we need per object to achieve even distribution
    prompts_per_object = main_object_count // n_objects
    remaining_prompts = main_object_count % n_objects

    # Initialize list to store sampled prompts
    sampled_prompts = []

    # For each object, sample prompts while trying to maintain even style distribution
    for i, obj in enumerate(unique_objects):
        # Add one extra prompt to the first 'remaining_prompts' objects to account for division remainder
        n_samples = prompts_per_object + (1 if i < remaining_prompts else 0)

        # Get all prompts for this object
        obj_df = other_objects_df[other_objects_df["object"] == obj]

        # Calculate how many prompts we want per style for this object
        prompts_per_style = n_samples // n_styles
        remaining_style_prompts = n_samples % n_styles

        # Sample prompts for each style
        style_samples = []
        for j, style in enumerate(unique_styles):
            # Add one extra prompt to the first 'remaining_style_prompts' styles
            style_n_samples = prompts_per_style + (
                1 if j < remaining_style_prompts else 0
            )

            # Get prompts for this style within this object
            style_df = obj_df[obj_df["style"] == style]

            # If we have more prompts than needed, sample randomly
            if len(style_df) > style_n_samples:
                style_df = style_df.sample(n=style_n_samples, random_state=random_state)
            elif len(style_df) < style_n_samples:
                # If we don't have enough prompts for this style, take all of them
                pass

            style_samples.append(style_df)

        # Combine all style samples for this object
        obj_sampled = pd.concat(style_samples, ignore_index=True)

        # If we still need more prompts to reach n_samples, sample from remaining prompts
        if len(obj_sampled) < n_samples:
            remaining_df = obj_df[~obj_df.index.isin(obj_sampled.index)]
            if len(remaining_df) > 0:
                additional_samples = remaining_df.sample(
                    n=min(n_samples - len(obj_sampled), len(remaining_df)),
                    random_state=random_state,
                )
                obj_sampled = pd.concat(
                    [obj_sampled, additional_samples], ignore_index=True
                )

        sampled_prompts.append(obj_sampled)

    # Combine main object and sampled other objects
    result_df = pd.concat([main_object_df] + sampled_prompts, ignore_index=True)

    return result_df


# %%

df = pd.read_parquet(
    "/workspace/sae/src/data/activation_capture_prompts/all_prompts.parquet"
)
subsampled_df = subset_prompts(df, "Dogs", random_state=42)

# %%
print("Original DataFrame length:", len(df))
print("Subsampled DataFrame length:", len(subsampled_df))
print("\nCount of each object in original DataFrame:")
print(df["object"].value_counts())
# %%
print("\nCount of each object in subsampled DataFrame:")
print(subsampled_df["object"].value_counts())

# %%
# Analyze and visualize style distribution in non-dog portion
import matplotlib.pyplot as plt

non_dog_df = subsampled_df[subsampled_df["object"] != "Dogs"]
style_counts = non_dog_df["style"].value_counts()

# Create the plot
plt.figure(figsize=(15, 8))
plt.bar(range(len(style_counts)), style_counts.values.tolist())
plt.xticks(
    range(len(style_counts)), style_counts.index.tolist(), rotation=45, ha="right"
)
plt.title("Distribution of Styles in Non-Dog Portion of Dataset")
plt.xlabel("Style")
plt.ylabel("Number of Prompts")
plt.tight_layout()

# Print summary statistics
print("\nStyle distribution summary:")
print(f"Total number of styles: {len(style_counts)}")
print(f"Average prompts per style: {len(non_dog_df) / len(style_counts):.2f}")
print(f"Min prompts per style: {style_counts.min()}")
print(f"Max prompts per style: {style_counts.max()}")

# %%
