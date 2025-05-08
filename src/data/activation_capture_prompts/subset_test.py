# %%
from data.activation_capture_prompts.prepare import (
    balance_concepts_styles,
    load_generated_prompts,
)

# %%

df = load_generated_prompts()
subsampled_df = balance_concepts_styles(df, "Dogs", random_state=42)

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
