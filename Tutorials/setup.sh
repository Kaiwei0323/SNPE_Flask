#!/bin/bash

# Define directories
DOWNLOAD_DIR="/home/aim/Documents"
VIDEO_DIR="/home/aim/Videos"
ZIP_FILE="v2.26.0.240828.zip"
APP_DIR="/home/aim/Documents/SNPE_Flask/Tutorials"

# Create the necessary directories if they do not exist
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$VIDEO_DIR"

# Download the zip file
echo "Downloading SDK zip file..."
curl -L -o "$DOWNLOAD_DIR/$ZIP_FILE" "https://huggingface.co/datasets/kaiwei0323/my-sdk/resolve/main/v2.26.0.240828.zip"

# Check if the zip file exists before attempting to unzip
if [ -f "$DOWNLOAD_DIR/$ZIP_FILE" ]; then
  echo "Extracting zip file..."
  unzip "$DOWNLOAD_DIR/$ZIP_FILE" -d "$DOWNLOAD_DIR"
  echo "SDK extracted successfully."

  # Remove the zip file after extraction
  echo "Cleaning up..."
  rm "$DOWNLOAD_DIR/$ZIP_FILE"
  echo "ZIP file removed."
else
  echo "Error: ZIP file not found at $DOWNLOAD_DIR/$ZIP_FILE. Skipping extraction."
fi

# Download the video files into the correct directory
echo "Downloading video files..."
curl -L -o "$VIDEO_DIR/brain_tumor.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/brain_tumor.mp4"
curl -L -o "$VIDEO_DIR/fall.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/fall.mp4"
curl -L -o "$VIDEO_DIR/freeway.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/freeway.mp4"
curl -L -o "$VIDEO_DIR/med_ppe.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/med_ppe.mp4"
curl -L -o "$VIDEO_DIR/ppe.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/ppe.mp4"

echo "Video files downloaded successfully to $VIDEO_DIR"

apt --fix-broken install -y

# Update and install necessary system packages
echo "Updating package list..."
apt update -y

echo "Installing software-properties-common..."
apt install software-properties-common -y

echo "Adding the deadsnakes PPA for Python 3.10..."
add-apt-repository ppa:deadsnakes/ppa -y
apt update -y

echo "Installing Python 3.10 and other dependencies..."
apt install python3.10 python3.10-venv python3.10-dev -y

echo "Installing CMake..."
apt install cmake -y

echo "Installing Mosquitto and Mosquitto clients..."
apt install mosquitto mosquitto-clients -y

echo "Installing libcairo2-dev and libgirepository1.0-dev..."
apt-get install libcairo2-dev -y
apt-get install libgirepository1.0-dev -y

echo "Installing portaudio19-dev..."
apt install portaudio19-dev -y

# Install pip for Python 3.10
echo "Installing pip for Python 3.10..."
python3.10 "$APP_DIR/get-pip.py"

# Install dependencies from requirements.txt
echo "Installing Python dependencies from requirements.txt..."
python3.10 -m pip install -r "$APP_DIR/requirements.txt"

# Install k3s (lightweight Kubernetes)
echo "Installing k3s..."
curl -sfL https://get.k3s.io | sh -s - --flannel-backend=host-gw

# Export XDG_RUNTIME_DIR permanently by adding it to .bashrc
echo "Setting XDG_RUNTIME_DIR permanently..."
grep -qxF 'export XDG_RUNTIME_DIR=/run/user/0' ~/.bashrc || echo 'export XDG_RUNTIME_DIR=/run/user/0' >> ~/.bashrc

source ~/.bashrc

# Download the .onnx files
echo "Downloading ONNX models..."
curl -L -o "$APP_DIR/wav2vec2-large-xlsr-53-english.onnx" "https://huggingface.co/datasets/kaiwei0323/wav2vec2-onnx/resolve/main/wav2vec2-large-xlsr-53-english.onnx?download=true"
curl -L -o "$APP_DIR/wav2vec2-large-xlsr-53-english.quant.onnx" "https://huggingface.co/datasets/kaiwei0323/wav2vec2-onnx/resolve/main/wav2vec2-large-xlsr-53-english.quant.onnx?download=true" 

# Apply Kubernetes manifests with validation disabled
echo "Applying Kubernetes manifests..."
kubectl apply -f https://raw.githubusercontent.com/Edgenesis/shifu/v0.57.0/pkg/k8s/crd/install/shifu_install.yml --validate=false

echo "Setup complete!"

