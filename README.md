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
chmod +777 -R SNPE_Flask
```

3. Navigate to Project Directory
```
cd SNPE_Flask/Tutorials
```

4. Environment Setup (Take approximate 10 mins)
```
chmod +x setup.sh
./setup.sh
source ~/.bashrc
```

5. Kubernetes Sensor Setup
* Log into [Shifu Cloud](https://cloud.shifu.dev/#/user/login) and setup sensors.
Reference: https://github.com/Kaiwei0323/Kubernetes-Shifu-Installation-Guide

6. Run Application
```
python3.10 app.py
```

7. Demo Output

![Screenshot from 2025-02-07 22-34-21](https://github.com/user-attachments/assets/4b77b4ee-b454-4324-86ec-5f2ef95e984e)

8. MQTT Setup (Optional)
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
