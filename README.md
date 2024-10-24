# SNPE YOLO Setup Guide

## Prerequisites

### Hardware Platform:
- **QCS6490**
  - CPU: Octa-Core Kryo 670 
  - GPU: Qualcomm Adreno 643

### Operating System:
- **Ubuntu 20.04**

### SNPE SDK Version:
- **v2.26.0.240828**

### Supported Models:
- DETR_Resnet101, YOLOv8

### Dependencies:
- Python3.10
- pybind11
- cmake
- OpenCV
- Torch, Torchvision, Torchaudio
- Pillow
- matplotlib
- Flask
---

## SNPE SDK Installation

Download the Neural Processing SDK from [Qualcomm SNPE SDK](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai).

### Directory Structure
```
Documents
└── SNPE_Flask
└── v2.26.0.240828
```

## Setup Steps
1. Switch to Admin Mode
```
su
Password: oelinux123
```

2. Navigate to Project Directory
```
cd SNPE_Flask/Tutorials
```

3. Environment Setup
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
```

4. Run Application
```
python3.10 app.py
```

5. Demo Output

![Screenshot 2024-10-20 at 05-04-20 Camera Stream](https://github.com/user-attachments/assets/2b8bd0a0-d292-4647-bd96-c97caa0bb825)

