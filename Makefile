SHELL := /bin/bash

export PYTHONPATH := .

.PHONY: all install download_ckpts gather_activations generate_images tmux_run tmux_attach source_env assemble_prompts


INSTALL_DIR := $(HOME)/google-cloud-sdk
PIPE_PATH := CompVis/stable-diffusion-v1-4
SESSION_NAME := saeuron-pipeline
SAE_ACT_DIR := sae_activations
GENERATED_IMGS_DIR := generated_imgs
ARTIFACT_BUCKET := gs://image-steering-artifacts

# Run Config
SAE_CHECKPOINT := sae-ckpts/sae_stable-diffusion-v1-4/patch_topk_expansion_factor32_k32_multi_topkFalse_auxk_alpha0.0_stable-diffusion-v1-4
HOOKPOINT := unet.up_blocks.1.attentions.1
NUM_CONCEPTS := 20
PROMPTS_PER_CONCEPT := 80
THEMES_PER_PROMPT_GATHER := 9
THEMES_PER_PROMPT_GENERATE := 2
STEPS := 80


# Default target to run everything
all: source_env download_ckpts gather_activations generate_images

# Run all tasks in a tmux session
tmux_run:
	tmux new-session -s $(SESSION_NAME) 'source .venv/bin/activate; make generate_images; echo "Pipeline completed. Press any key to exit."; read'
	@echo "Started pipeline in tmux session named '$(SESSION_NAME)'"
	@echo "To attach to the session, run: tmux attach-session -t $(SESSION_NAME)"

# Attach to existing tmux session
tmux_attach:
	tmux attach-session -t $(SESSION_NAME)

# Install dependencies
install:
	mkdir -p $(INSTALL_DIR)
	apt-get update && apt-get install -y tmux
	curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
	tar -xf google-cloud-cli-linux-x86_64.tar.gz -C $(INSTALL_DIR) --strip-components=1
	$(INSTALL_DIR)/install.sh
	rm google-cloud-cli-linux-x86_64.tar.gz
	$(INSTALL_DIR)/bin/gcloud init

source_env:
	source .venv/bin/activate

assemble_prompts: 
	python data/activation_capture_prompts/prepare.py

save_diffusion_activations:
	python src/save_activations_runner.py \
		--max_prompts_per_concept 1 \


# Sync checkpoint data
download_ckpts: 
	gsutil -m cp -r gs://sae-ckpts/sae-ckpts/ .
	
# Map Concepts to SAE Latents
save_latents_per_concept: download_ckpts
	python src/save_latents_per_concept.py \
		--checkpoint_path $(SAE_CHECKPOINT) \
		--hookpoint "$(HOOKPOINT)" \
		--pipe_path "$(PIPE_PATH)" \
		--save_dir $(SAE_ACT_DIR) \
		--num_concepts $(NUM_CONCEPTS) \
		--prompts_per_concept $(PROMPTS_PER_CONCEPT) \
		--themes_per_prompt $(THEMES_PER_PROMPT_GATHER) \
		--steps $(STEPS)

# Generate Images
generate_images: save_latents_per_concept
	python src/generate_images.py \
		--percentiles [99.99,99.995,99.999] \
		--multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0] \
		--seed 42 \
		--sae_checkpoint $(SAE_CHECKPOINT) \
		--pipe_checkpoint "$(PIPE_PATH)" \
		--hookpoint '$(HOOKPOINT)' \
		--class_latents_path $(SAE_ACT_DIR)/concept_latents_dict_$(HOOKPOINT).pkl \
		--output_dir $(GENERATED_IMGS_DIR) \
		--num_concepts $(NUM_CONCEPTS) \
		--prompts_per_concept $(PROMPTS_PER_CONCEPT) \
		--themes_per_prompt $(THEMES_PER_PROMPT_GENERATE) \
		--steps $(STEPS)

upload_generate_imgs_artifacts: 
	chmod +x upload_artifacts.sh
	./upload_artifacts.sh \
		--pipe-path "$(PIPE_PATH)" \
		--session-name "$(SESSION_NAME)" \
		--sae-checkpoint "$(SAE_CHECKPOINT)" \
		--sae-act-dir "$(SAE_ACT_DIR)" \
		--generated-imgs-dir "$(GENERATED_IMGS_DIR)" \
		--artifact-bucket "$(ARTIFACT_BUCKET)" \
		--hookpoint "$(HOOKPOINT)" \
		--num-concepts "$(NUM_CONCEPTS)" \
		--prompts-per-concept "$(PROMPTS_PER_CONCEPT)" \
		--themes-per-prompt-gather "$(THEMES_PER_PROMPT_GATHER)" \
		--themes-per-prompt-generate "$(THEMES_PER_PROMPT_GENERATE)" \
		--steps "$(STEPS)"
	
	