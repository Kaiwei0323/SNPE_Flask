# SNPE FLASK Setup Guide

## Prerequisites

### Hardware Requirements
- Platform: **QCS6490**
  - CPU: Octa-Core Kryo 670 
  - GPU: Qualcomm Adreno 643

### Software Requirements
- Operating System: **Ubuntu 20.04 (arm64)**
- SNPE SDK Version: **v2.26.0.240828**
- Supported Models: DETR_Resnet101, YOLOv8, YOLOv11

### Dependencies:
- Python3.10
- pybind11
- cmake
- OpenCV
- Torch, Torchvision, Torchaudio
- Pillow
- matplotlib
- Flask
- paho-mqtt
- mosquitto mosquitto-clients
- pygobject
---

## SNPE SDK Installation
* v2.26.0.240828
```
curl -L -O "https://huggingface.co/datasets/kaiwei0323/my-sdk/resolve/main/v2.26.0.240828.zip"
```
Download the Neural Processing SDK from [Qualcomm SNPE SDK](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai).

## Download Pre-recorded Videos
* Download Link

```
curl -L -O "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/brain_tumor.mp4"
```
```
curl -L -O "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/fall.mp4"
```
```
curl -L -O "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/freeway.mp4"
```
```
curl -L -O "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/med_ppe.mp4"
```
```
curl -L -O "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/ppe.mp4"
```

## Directory Structure
```
Documents
└── SNPE_Flask
└── v2.26.0.240828
Videos
└── freeway.mp4
└── ppe.mp4
└── fall.mp4
└── brain_tumor.mp4
└── med_ppe.mp4
```

## Setup Steps
1. Switch to Admin Mode
```
su
Password: oelinux123
```

2. Clone and Install SNPE_Flask Project
```
apt install git
git clone https://github.com/Kaiwei0323/SNPE_Flask.git -b demo
```

3. Navigate to Project Directory
```
cd SNPE_Flask/Tutorials
```

4. Environment Setup
```
apt update
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.10 python3.10-venv python3.10-dev
python3.10 get-pip.py
python3.10 -m pip install pybind11
apt install cmake
python3.10 -m pip install opencv-python
pip install tqdm
python3.10 -m pip install torch torchvision torchaudio
python3.10 -m pip install Pillow
python3.10 -m pip install matplotlib
python3.10 -m pip install Flask --ignore-installed blinker
python3.10 -m pip install paho-mqtt
apt install mosquitto mosquitto-clients
apt-get install libcairo2-dev
apt-get install libgirepository1.0-dev
python3.10 -m pip install pygobject==3.50.0
python3.10 -m pip install scikit-learn streamlit==1.31.1 scikit-learn==1.3.2 joblib pandas
apt install portaudio19-dev
python3.10 -m pip install pyaudio
python3.10 -m pip install psutil
curl -sfL https://get.k3s.io | sh -s - --flannel-backend=host-gw
```

5. Install wav2vec2 ONNX Model
```
wget "https://huggingface.co/datasets/kaiwei0323/wav2vec-onnx/resolve/main/wav2vec2-large-xlsr-53-english.onnx?download=true" -O wav2vec2-large-xlsr-53-english.onnx
wget "https://huggingface.co/datasets/kaiwei0323/wav2vec-onnx/resolve/main/wav2vec2-large-xlsr-53-english_quant.onnx?download=true" -O wav2vec2-large-xlsr-53-english_quant.onnx
```

6. Kubernetes Sensor Setup
```
kubectl apply -f https://raw.githubusercontent.com/Edgenesis/shifu/v0.57.0/pkg/k8s/crd/install/shifu_install.yml

kubectl apply -f 'https://cloud.shifu.dev/manifests/1fc51d1d-fa77-4721-9765-f2e0e9b2acb1.yaml?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=root%2F20250414%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250414T200321Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=98aaf0ad0f9cb89028845a4d2470aef7763ca1cb5d9b21bc536fb741da62ed6a'

kubectl apply -f 'https://cloud.shifu.dev/manifests/cdfc4158-ee27-4816-bdae-6c7467eb0999.yaml?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=root%2F20250414%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250414T200340Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=a34e8ada0d469b7ce4620ad1dfd08c97c6c82d0cbd169e80e2890bf0518b97fa'
```

7. Run Application
```
python3.10 app.py
```

8. Demo Output

![Screenshot from 2025-02-07 22-34-21](https://github.com/user-attachments/assets/4b77b4ee-b454-4324-86ec-5f2ef95e984e)

9. MQTT Setup (Optional)
Enable and check the Mosquitto service
```
systemctl enable mosquitto
systemctl status mosquitto
```
Subscribe to detection topics:
* For detection time
```
mosquitto_sub -h localhost -t detection_time -v
```
* For YOLOv8 detection:
```
mosquitto_sub -h localhost -t yolov8/detections -v
```
* For DETR detection:
```
mosquitto_sub -h localhost -t detr/detections -v
```
