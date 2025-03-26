# SNPE FLASK Setup Guide

## Prerequisites

### Hardware Requirements
- Platform: **QCS6490**
  - CPU: Octa-Core Kryo 670 
  - GPU: Qualcomm Adreno 643

### Software Requirements
- Operating System: **Ubuntu 20.04 (arm64)**
- SNPE SDK Version: **v2.26.0.240828**
- Supported Models: **DETR**, **YOLOv8**, **YOLOv11**, **YOLOv12**

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
Extract the file and place it in the Documents folder.
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
### 1. Switch to Admin Mode
```
su
Password: oelinux123
```

### 2. Clone and Install SNPE_Flask Project
```
apt install git
git clone https://github.com/Kaiwei0323/SNPE_Flask.git
```

### 3. Navigate to Project Directory
```
cd SNPE_Flask/Tutorials
```

### 4. Environment Setup
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

### 5. Run Application
```
python3.10 app.py
```

### 6. Demo Output

![Screenshot from 2024-11-20 22-27-25](https://github.com/user-attachments/assets/48dd959c-8b56-4b08-a4f8-f379255f2386)

### 7. Sample Input
* Camera Name: Demo
* Video Source: RTSP
* RTSP URL: rtsp://99.64.152.69:8554/mystream2
* Model: YOLOV8S_DSP
* Runtime: DSP
  
**Note:**
* Models with the suffix "_DSP" are designed to run exclusively on the DSP runtime.
* Models with the suffix "_GPU" can run on both CPU and GPU.


### 8. MQTT Setup (Optional)

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

## Deploy your own model
### 1. Convert Your Model to .dlc Format
* Visit our Model Conversion website: [Model Conversion Website](http://99.64.152.69:5000/). 
* Go to **Model Conversion** Tab.
* Refer to the Application User Manual Section for detailed instructions on how to convert your model to the .dlc format: [User Manual](https://github.com/Kaiwei0323/qc_model_conversion_flask).

### 2. Visualize Your Model
* After conversion, use the Model Visualization tab on the website to visualize your model.
* Find and note the input layer and output layer names of your model.
![Screenshot from 2025-03-06 21-44-16](https://github.com/user-attachments/assets/45f9f79c-5a94-4171-8b1b-c22c67806705)



### 3. Add Your Model to the Project
* Place your .dlc model file in the SNPE_Flask/Tutorials/models/ folder.
* Create a Python class file for your model and save it in the SNPE_Flask/Tutorials/myclasses/ folder.
* Update the __init__.py file inside myclasses/

### 4. Modify the camera.py File
* Open the SNPE_Flask/Tutorials/camera.py file.
* Modify the model_map (lines 24-35) to include your new model. This will ensure that the application can recognize and use your model.
* In the example above, add **"YOLOV8S_DSP": ("models/yolov8s_encode_int8.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], COCO80_CLASSES)** to the model_map.

### 5. Run the Application
* After completing the above steps, rerun the application. Your model will now be available for selection and use within the app.
