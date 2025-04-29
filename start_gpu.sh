# Load environment variables
source .env

git clone https://${GITHUB_TOKEN}@github.com/yash-shirsath/gradient-routing.git
git config --global user.email "yash.shirsath@gmail.com"
git config --global user.name "Yash Shirsath (GPU)"


touch ~/.no_auto_tmux
pip uninstall torch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# in parallel
sudo apt install build-essential

cursor --install-extension ms-python.python
cursor --install-extension nvidia.nsight-vscode-edition


pip install transformers datasets jaxtyping wandb
pip install -e .
cd steering/ 

wandb login ${WANDB_API_KEY}

#gcloud 
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

gcloud init
cd '/root/gradient-routing/steering/data/fineweb-edu'
gsutil -m cp gs://gradient-routing/*.bin .


# Materialize the script with actual tokens
source .env && envsubst '${GITHUB_TOKEN} ${WANDB_API_KEY}' < start_gpu.sh > start_gpu_materialized.sh