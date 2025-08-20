#!/bin/bash

# Define directories
DOWNLOAD_DIR="/home/aim/Documents"
VIDEO_DIR="/home/aim/Videos"
ZIP_FILE="v2.26.0.240828.zip"

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
  # Delete the zip file after extraction
  rm "$DOWNLOAD_DIR/$ZIP_FILE"
  echo "ZIP file deleted."
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
python3.10 get-pip.py

# Install dependencies from requirements.txt
echo "Installing Python dependencies from requirements.txt..."
python3.10 -m pip install -r requirements.txt

# Set up DSP environment variables and add to .bashrc
echo "Setting up DSP environment variables..."

# Define SNPE paths
SNPE_ROOT="/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828"
TUTORIALS_DIR="/home/aim/Documents/SNPE_Flask/Tutorials"

# Create the DSP environment configuration
cat >> ~/.bashrc << 'EOF'

# SNPE DSP Environment Variables
export SNPE_ROOT="/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828"
export ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export HEXAGON_ARM_SYSROOT="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_LIBRARY_PATH="$SNPE_ROOT/lib/aarch64-ubuntu-gcc9.4"
export SNPE_HEXAGON_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_APP_DIR="/home/aim/Documents/SNPE_Flask/Tutorials"

EOF

# Source the updated .bashrc for current session
source ~/.bashrc

echo "DSP environment variables added to .bashrc"
echo "SNPE_ROOT: $SNPE_ROOT"
echo "ADSP_LIBRARY_PATH: $ADSP_LIBRARY_PATH"

echo "Setup complete!"

