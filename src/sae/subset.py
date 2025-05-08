# %%
from typing import Optional

import pandas as pd

from data.activation_capture_prompts.prepare import load_generated_prompts


def balance_prompts_objects_styles(
    df: pd.DataFrame, main_concept: str, random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Subset the prompts DataFrame to create a balanced dataset with a main concept and other concepts.
    Ensures even distribution of both concepts and styles in the non-main concept portion.

    Args:
        df: Input DataFrame containing prompts with 'concept' and 'style' columns
        main_concept: The main concept to include all prompts for
        random_state: Optional random seed for reproducibility

    Returns:
        DataFrame containing:
        - All prompts for the main concept
        - An even distribution of other concepts and their styles
    """
    # Get all prompts for the main concept
    main_concept_df = df[df["concept"] == main_concept].copy()
    main_concept_count = len(main_concept_df)

    # Get all other concepts
    other_concepts_df = df[df["concept"] != main_concept].copy()

    # Get unique concepts and styles
    unique_concepts = other_concepts_df["concept"].unique()
    unique_styles = other_concepts_df["style"].unique()
    n_concepts = len(unique_concepts)
    n_styles = len(unique_styles)

    # Calculate how many prompts we need per concept to achieve even distribution
    prompts_per_concept = main_concept_count // n_concepts
    remaining_prompts = main_concept_count % n_concepts

    # Initialize list to store sampled prompts
    sampled_prompts = []

    # For each concept, sample prompts while trying to maintain even style distribution
    for i, concept in enumerate(unique_concepts):
        # Add one extra prompt to the first 'remaining_prompts' concepts to account for division remainder
        n_samples = prompts_per_concept + (1 if i < remaining_prompts else 0)

        # Get all prompts for this concept
        concept_df = other_concepts_df[other_concepts_df["concept"] == concept]

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

            # Get prompts for this style within this concept
            style_df = concept_df[concept_df["style"] == style]

            # If we have more prompts than needed, sample randomly
            if len(style_df) > style_n_samples:
                style_df = style_df.sample(n=style_n_samples, random_state=random_state)
            elif len(style_df) < style_n_samples:
                # If we don't have enough prompts for this style, take all of them
                pass

            style_samples.append(style_df)

        # Combine all style samples for this concept
        concept_sampled = pd.concat(style_samples, ignore_index=True)

        # If we still need more prompts to reach n_samples, sample from remaining prompts
        if len(concept_sampled) < n_samples:
            remaining_df = concept_df[~concept_df.index.isin(concept_sampled.index)]
            if len(remaining_df) > 0:
                additional_samples = remaining_df.sample(
                    n=min(n_samples - len(concept_sampled), len(remaining_df)),
                    random_state=random_state,
                )
                concept_sampled = pd.concat(
                    [concept_sampled, additional_samples], ignore_index=True
                )

        sampled_prompts.append(concept_sampled)

    # Combine main concept and sampled other concepts
    result_df = pd.concat([main_concept_df] + sampled_prompts, ignore_index=True)

    return result_df


# %%

df = load_generated_prompts()
subsampled_df = balance_prompts_objects_styles(df, "Dogs", random_state=42)

# %%
print("Original DataFrame length:", len(df))
print("Subsampled DataFrame length:", len(subsampled_df))
print("\nCount of each object in original DataFrame:")
print(df["concept"].value_counts())
# %%
print("\nCount of each object in subsampled DataFrame:")
print(subsampled_df["concept"].value_counts())

# %%
# Analyze and visualize style distribution in non-dog portion
import matplotlib.pyplot as plt

non_dog_df = subsampled_df[subsampled_df["concept"] != "Dogs"]
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
