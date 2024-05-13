#!/bin/bash

# Set up Miniconda for managing 

# If conda not installed 

echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init
conda update -y conda


# Create a new Conda environment with Python 3.10
echo "Creating a new Conda environment with Python 3.10..."
conda create -y -n ccs python=3.10

# Activate the environment
echo "Activating the environment..."
conda activate ccs

# Install Python libraries
echo "Installing Python libraries..."
pip install numpy pandas torch datasets transformers scikit-learn sentencepiece tqdm accelerate matplotlib

# Update and upgrade Ubuntu packages
echo "Updating Ubuntu packages..."
apt update && apt upgrade -y

# Add the Graphics Drivers PPA for latest NVIDIA drivers
echo "Adding the Graphics Drivers PPA..."
add-apt-repository ppa:graphics-drivers/ppa
apt update

# Automatically install the best drivers
echo "Installing NVIDIA drivers..."
ubuntu-drivers autoinstall

code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter 
code --install-extension ms-toolsai.jupyter-keymap
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension GitHub.copilot

curl -sS https://webi.sh/gh | sh

gh auth login

huggingface-cli login

TMP_DIR_PATH="/workspace/tmp"
HF_HOME_PATH="/workspace/.cache/huggingface"

echo "export TMPDIR=\"$TMP_DIR_PATH\"" >> ~/.bashrc
echo "export HF_HOME=\"$HF_HOME_PATH\"" >> ~/.bashrc
echo "export PATH=$PATH:~/miniconda3/bin" >> ~/.bashrc
echo "conda activate ccs" >> ~/.bashrc

source ~/.bashrc
