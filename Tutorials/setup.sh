#!/bin/bash

# Define directories
SDK_DIR="/data/sdk"
DOWNLOAD_DIR="/home/ubuntu/Documents"
VIDEO_DIR="/data/video"
ZIP_FILE="v2.26.0.240828.zip"

# Create the necessary directories if they do not exist
sudo mkdir -p "$SDK_DIR"
sudo mkdir -p "$VIDEO_DIR"

# Download the zip file
echo "Downloading SDK zip file..."
curl -L -o "$SDK_DIR/$ZIP_FILE" "https://huggingface.co/datasets/kaiwei0323/my-sdk/resolve/main/v2.26.0.240828.zip"

# Check if the zip file exists before attempting to unzip
if [ -f "$SDK_DIR/$ZIP_FILE" ]; then
  echo "Extracting zip file..."
  unzip "$SDK_DIR/$ZIP_FILE" -d "$SDK_DIR"
  echo "SDK extracted successfully."
  # Delete the zip file after extraction
  rm "$SDK_DIR/$ZIP_FILE"
  echo "ZIP file deleted."
else
  echo "Error: ZIP file not found at $SDK_DIR/$ZIP_FILE. Skipping extraction."
fi

# Download the video files into the correct directory
echo "Downloading video files..."
curl -L -o "$VIDEO_DIR/brain_tumor.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/brain_tumor.mp4"
curl -L -o "$VIDEO_DIR/fall.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/fall.mp4"
curl -L -o "$VIDEO_DIR/freeway.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/freeway.mp4"
curl -L -o "$VIDEO_DIR/med_ppe.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/med_ppe.mp4"
curl -L -o "$VIDEO_DIR/ppe.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/ppe.mp4"

echo "Video files downloaded successfully to $VIDEO_DIR"

# Set up DSP environment variables and add to .bashrc
echo "Setting up DSP environment variables..."

# Define SNPE paths
SNPE_ROOT="$SDK_DIR/v2.26.0.240828/qairt/2.26.0.240828"
TUTORIALS_DIR="$DOWNLOAD_DIR/SNPE_Flask/Tutorials"

# Create the DSP environment configuration
cat >> ~/.bashrc << 'EOF'

# SNPE DSP Environment Variables
export SNPE_ROOT="/data/sdk/v2.26.0.240828/qairt/2.26.0.240828"
export ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export HEXAGON_ARM_SYSROOT="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_LIBRARY_PATH="$SNPE_ROOT/lib/aarch64-ubuntu-gcc9.4"
export SNPE_HEXAGON_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_APP_DIR="$TUTORIALS_DIR"

EOF

# Source the updated .bashrc for current session
source ~/.bashrc

echo "DSP environment variables added to .bashrc"
echo "SNPE_ROOT: $SNPE_ROOT"
echo "ADSP_LIBRARY_PATH: $ADSP_LIBRARY_PATH"

echo "Setting up dependencies..."

# Add the repository for Qualcomm IoT apps
sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-ppa

# Update package lists
sudo apt update -y

# Install necessary packages
sudo apt install -y \
    gstreamer1.0-qcom-sample-apps \
    python3-pip \
    python3-pybind11 \
    cmake \
    python3-flask \
    python3-opencv \
    python3-paho-mqtt \
    python3-gi \
    python3-gst-1.0 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    mosquitto \
    mosquitto-clients

# Start and enable mosquitto service
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# Install Python packages
sudo python3 -m pip install --break-system-packages torch
sudo python3 -m pip install --break-system-packages torchvision

echo "Setup complete!"

