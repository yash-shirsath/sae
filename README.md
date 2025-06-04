# Source Code for Steering Stable Diffusion Using Sparse Autoencoderes

## Structure

### ./

contains scripts used to run all steps of the steering pipeline. see @Makefile for more details.

1. Capture Diffusion Activations
1. Train SAE
1. Map Concepts to SAE Latents
1. Steer Images
1. Upload artifacts

### ./sae/

contains sae model and training implementations as well as hooked implementations of HuggingFace's diffusion pipeline

## Acknowledgements

code adapted from

- Cywiński, B., & Deja, K. (2025). SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders. arXiv:2501.18052v2 [cs.LG].
- Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093 [cs.LG].
