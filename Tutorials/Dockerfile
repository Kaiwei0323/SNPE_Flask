# =========================================
# Stage 1: Build Stage - Optional Compilation of libsnpehelper.so
# =========================================
FROM --platform=arm64 ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    make \
    g++ \
    gcc \
    python3.12 \
    python3.12-dev \
    python3-pip \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Configure Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pybind11
RUN pip3 install --break-system-packages pybind11

WORKDIR /app

# Copy source code
COPY snpehelper/ /app/snpehelper/

# Placeholder SDK path
RUN mkdir -p /data/sdk

# Try to build libsnpehelper.so (optional)
WORKDIR /app/snpehelper
RUN mkdir -p build && cd build && \
    (cmake .. && make 2>&1 || echo "Build skipped - using pre-built .so from Tutorials") || true

# =========================================
# Stage 2: Runtime Stage - Flask + SNPE + GStreamer + MQTT + Torch
# =========================================
FROM --platform=arm64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ---- Install runtime dependencies ----
# Install software-properties-common for add-apt-repository
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Qualcomm IoT repository
RUN add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-ppa

# Install runtime dependencies
# Note: The QMMF library (libqmmf_recorder_client.so) warning is non-fatal.
# The plugin will be skipped if the library is not available, but the app will still work.
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    python3-gi \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    libopencv-dev \
    python3-opencv \
    python3-pybind11 \
    python3-flask \
    python3-paho-mqtt \
    gstreamer1.0-qcom-sample-apps \
    mosquitto \
    mosquitto-clients \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

# ---- Copy the app files ----
COPY Tutorials/ /app/

# ---- Set SNPE environment variables ----
ENV SNPE_ROOT=/data/sdk/v2.26.0.240828/qairt/2.26.0.240828
ENV ADSP_LIBRARY_PATH=${SNPE_ROOT}/lib/hexagon-v68/unsigned
ENV HEXAGON_ARM_SYSROOT=${SNPE_ROOT}/lib/hexagon-v68/unsigned
ENV SNPE_LIBRARY_PATH=${SNPE_ROOT}/lib/aarch64-ubuntu-gcc9.4
ENV SNPE_HEXAGON_LIBRARY_PATH=${SNPE_ROOT}/lib/hexagon-v68/unsigned
ENV SNPE_APP_DIR=/app
# Set LD_LIBRARY_PATH with SNPE libraries first (critical for DSP runtime)
# Order matters: SNPE libraries must come before system libraries
# Include QMMF library path (will be mounted from host at /usr/lib/qcm6490)
ENV LD_LIBRARY_PATH=${SNPE_LIBRARY_PATH}:${ADSP_LIBRARY_PATH}:/usr/lib/qcm6490:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/app:${PYTHONPATH}
# Ensure DSP libraries are accessible
ENV PATH=${PATH}:${SNPE_ROOT}/bin/x86_64-linux-clang:${SNPE_ROOT}/bin/aarch64-ubuntu-gcc9.4

# ---- Install Python requirements ----
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed -r requirements.txt || true

# ---- Install PyTorch and torchvision ----
RUN pip3 install --break-system-packages torch torchvision

# ---- Create necessary directories ----
RUN mkdir -p /data/sdk /data/video

# ---- Download video files ----
RUN curl -L -o /data/video/brain_tumor.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/brain_tumor.mp4" && \
    curl -L -o /data/video/fall.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/fall.mp4" && \
    curl -L -o /data/video/freeway.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/freeway.mp4" && \
    curl -L -o /data/video/med_ppe.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/med_ppe.mp4" && \
    curl -L -o /data/video/ppe.mp4 "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/ppe.mp4" && \
    echo "Video files downloaded successfully"

# ---- Expose Flask port ----
EXPOSE 5001

# ---- Health check ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5001')" || exit 1

# ---- Run mosquitto and Flask ----
CMD service mosquitto start && \
    python3 app.py --host=0.0.0.0 --port=5001

