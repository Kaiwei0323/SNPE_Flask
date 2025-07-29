# Use Ubuntu 20.04 as base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    unzip \
    cmake \
    mosquitto \
    mosquitto-clients \
    libcairo2-dev \
    libgirepository1.0-dev \
    gobject-introspection \
    portaudio19-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.10 and install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Upgrade pip and install build tools
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Install GObject and GStreamer development files and install for Python 3.10
RUN apt-get update && apt-get install -y \
    libgirepository1.0-dev \
    gobject-introspection \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && python3.10 -m pip install pygobject

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p /home/aim/Documents /home/aim/Videos

# Download and extract SNPE SDK
RUN cd /home/aim/Documents && \
    curl -L -o v2.26.0.240828.zip "https://huggingface.co/datasets/kaiwei0323/my-sdk/resolve/main/v2.26.0.240828.zip" && \
    unzip v2.26.0.240828.zip && \
    rm v2.26.0.240828.zip

# Download demo videos
RUN cd /home/aim/Videos && \
    curl -L -o brain_tumor.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/brain_tumor.mp4" && \
    curl -L -o fall.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/fall.mp4" && \
    curl -L -o freeway.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/freeway.mp4" && \
    curl -L -o med_ppe.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/med_ppe.mp4" && \
    curl -L -o ppe.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/ppe.mp4"

# Set SNPE environment variables
ENV SNPE_ROOT="/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828"
ENV ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
ENV HEXAGON_ARM_SYSROOT="$SNPE_ROOT/lib/hexagon-v68/unsigned"
ENV SNPE_LIBRARY_PATH="$SNPE_ROOT/lib/aarch64-ubuntu-gcc9.4"
ENV SNPE_HEXAGON_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
ENV SNPE_APP_DIR="/app"
ENV LD_LIBRARY_PATH="/app:$SNPE_LIBRARY_PATH:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/app:/usr/local/lib/python3.10/dist-packages:$PYTHONPATH"

# DSP-specific environment variables
ENV ADSP_LIBRARY_PATH="/media/aim/dsp/adsp"
ENV HEXAGON_ARM_SYSROOT="/media/aim/dsp/adsp"
ENV SNPE_HEXAGON_LIBRARY_PATH="/media/aim/dsp/adsp"
ENV QNN_SYSCACHE_PATH="/tmp/qnn_cache"
ENV QNN_LOG_LEVEL="info"
ENV LD_LIBRARY_PATH="/media/aim/dsp/adsp:$LD_LIBRARY_PATH"

# Set DSP runtime as default
ENV SNPE_DEFAULT_RUNTIME="DSP"

# Make the libsnpehelper.so executable if it exists and copy to system library path
RUN if [ -f /app/libsnpehelper.so ]; then \
        chmod +x /app/libsnpehelper.so && \
        cp /app/libsnpehelper.so /usr/local/lib/ && \
        ldconfig; \
    fi

# Run DSP setup and verification
RUN chmod +x /app/setup_dsp.sh && /app/setup_dsp.sh

# Expose port
EXPOSE 5001

# Set the default command
CMD ["python3.10", "app.py", "--host=0.0.0.0", "--port=5001"] 