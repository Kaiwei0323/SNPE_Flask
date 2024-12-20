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

Download the Neural Processing SDK from [Qualcomm SNPE SDK](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai).

## Download Pre-recorded Videos
1. Access the SFTP server
    - Use the credentials for the **qcs6490** user to access the server
2. Navigate to the video directory
    - The videos are located in: 
```
/Files/QC01U/Demo_Video
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
git clone https://github.com/Kaiwei0323/SNPE_Flask.git
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
python3.10 -m pip install --upgrade pygobject
```

5. MQTT Setup
Enable and check the Mosquitto service
```
systemctl enable mosquitto
systemctl status mosquitto
```
Subscribe to detection topics:
* For YOLOv8 detection:
```
mosquitto_sub -h localhost -t yolov8/detections -v
```
* For DETR detection:
```
mosquitto_sub -h localhost -t detr/detections -v
```

6. Run Application
```
python3.10 app.py
```

7. Demo Output

![Screenshot from 2024-11-20 22-27-25](https://github.com/user-attachments/assets/48dd959c-8b56-4b08-a4f8-f379255f2386)


