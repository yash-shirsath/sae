# Steering Stable Diffusion Using Sparse Autoencoders

This project implements a pipeline for steering Stable Diffusion models using Sparse Autoencoders (SAEs) to enable interpretable concept manipulation in generated images. The implementation is based on cutting-edge research in mechanistic interpretability and allows for fine-grained control over specific concepts in diffusion models.

## Overview

The pipeline consists of several key stages:

1. **Activation Capture**: Extract internal activations from Stable Diffusion during image generation
2. **SAE Training**: Train sparse autoencoders to learn interpretable feature representations
3. **Concept Mapping**: Identify which SAE latents correspond to specific visual concepts
4. **Image Generation**: Generate images with controlled concept steering

## Quick Start

### Prerequisites

- Python 3.10+
- Hopper GPU (recommended: H100 or similar)

### Installation

1. **Set up Python environment**:

   ```bash
   pip install uv
   uv venv
   source .venv/bin/activate
   uv sync
   ```

2. **Install system dependencies**:
   ```bash
   make install
   ```

### Running the Pipeline

Execute the complete pipeline using the following commands:

```bash
# 1. Prepare training prompts
make assemble_prompts

# 2. Capture diffusion model activations (~10 hours on H100, ~2.6TB Disk)
make save_diffusion_activations

# 3. Train sparse autoencoder (~4 hours on H100)
make train_sae

# 4. Map concepts to SAE latents (~1 hour on H100)
make save_latents_per_concept

# 5. Generate steered images (~8 hours on H100)
make generate_images
```

**Alternative**: Run the entire pipeline in a tmux session:

```bash
make tmux_run
```

## Project Structure

### Root Directory (`./`)

Contains high-level scripts for executing the complete steering pipeline, including Makefile configurations and orchestration scripts.

### SAE Module (`./sae/`)

Core implementation including:

- SAE model architecture and training logic
- Hooked HuggingFace diffusion pipeline for activation capture
- Utility functions for model manipulation and analysis

### Data Directory (`./data/`)

Contains:

- Sample prompts for training and evaluation
- Data preparation and preprocessing scripts
- Concept definitions and categorizations

## Acknowledgements

Code is adapted from:

- **Cywiński, B., & Deja, K.** (2025). _SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders_. arXiv:2501.18052v2 [cs.LG].

- **Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J.** (2024). _Scaling and evaluating sparse autoencoders_. arXiv:2406.04093 [cs.LG].

**Dataset**: Prompt data adapted from Zhang, Y., et al. (2024). _UnlearnCanvas: A Stylized Image Dataset for Enhanced Machine Unlearning Evaluation in Diffusion Models_.

## License

This project is released under the MIT License. See `LICENSE` file for details.
