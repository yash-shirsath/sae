# Load environment variables
source .env

git clone https://${GITHUB_TOKEN}@github.com/yash-shirsath/sae.git
git config --global user.email "yash.shirsath@gmail.com"
git config --global user.name "Yash Shirsath (GPU)"

touch ~/.no_auto_tmux

cursor --install-extension ms-python.python
cursor --install-extension ms-toolsai.jupyter
cursor --install-extension nvidia.nsight-vscode-edition
cursor --install-extension charliermarsh.ruff

pip install uv
uv venv
source .venv/bin/activate
pip install uv
uv sync

wandb login ${WANDB_API_KEY}

#gcloud 
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
gcloud init



# Materialize the script with actual tokens
source .env && envsubst '${GITHUB_TOKEN} ${WANDB_API_KEY}' < start_gpu.sh > start_gpu_materialized.sh


# archive
pip uninstall torch
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

sudo apt install build-essential

gsutil -m cp gs://gradient-routing/*.bin .
gsutil -m cp -r sae-ckpts/ gs://sae-ckpts/