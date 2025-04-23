# 🚀 SNPE Flask Setup Guide

> Vision solution using SNPE SDK for real-time inference, Kubernetes-wrapped sensors for smart farming, and Wav2Vec2 for voice recognition.

![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2020.04-blue?logo=ubuntu)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![SNPE](https://img.shields.io/badge/SNPE-v2.26.0.240828-red?logo=qualcomm)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Supported Models
- ✅ YOLOv8  
- ✅ YOLOv11  
- ✅ DETR  

---

## Hardware Requirements

| Component | Specification |
|----------|----------------|
| Platform | **QCS6490** |
| CPU      | Octa-Core Kryo 670 |
| GPU      | Qualcomm Adreno 643 |

---

## Software Requirements

- OS: Ubuntu 20.04 (arm64)
- SNPE SDK Version: **v2.26.0.240828**
- Python 3.10

---

## Directory Structure

```bash
Documents/
├── SNPE_Flask/
├── v2.26.0.240828/
Videos/
├── freeway.mp4
├── ppe.mp4
├── fall.mp4
├── brain_tumor.mp4
└── med_ppe.mp4
```

---

## Setup Steps

### 1. Switch to Admin Mode
```bash
su
Password: oelinux123
```

---

### 2. Clone the Project
```bash
apt install git
cd /home/aim/Documents
git clone https://github.com/Kaiwei0323/SNPE_Flask.git -b demo
chmod +777 -R SNPE_Flask
```

---

### 3. Navigate to the Project Directory
```bash
cd SNPE_Flask/Tutorials
```

---

### 4. Environment Setup (Takes ~10 minutes)
```bash
chmod +x setup.sh
./setup.sh
```

> 🔍 `setup.sh` installs dependencies, sets up SNPE paths, and configures the environment for Flask + SNPE.

---

### 5. Setup Kubernetes Sensors (Optional)

Log in to [Shifu Cloud](https://cloud.shifu.dev/#/user/login) and configure sensor devices.

📘 [Kubernetes-Shifu-Installation-Guide](https://github.com/Kaiwei0323/Kubernetes-Shifu-Installation-Guide)

---

### 6. Run the Application
```bash
python3.10 app.py
```

---

### 7. Demo Output

![Screenshot from 2025-02-07 22-34-21](https://github.com/user-attachments/assets/4b77b4ee-b454-4324-86ec-5f2ef95e984e)

---

### 8. MQTT Setup (Optional)

#### Enable Mosquitto Service
```bash
systemctl enable mosquitto
systemctl status mosquitto
```

#### Subscribe to Topics

##### Detection Time
```bash
mosquitto_sub -h localhost -t detection_time -v
```

##### YOLOv8 Detection
```bash
mosquitto_sub -h localhost -t yolov8/detections -v
```

##### DETR Detection
```bash
mosquitto_sub -h localhost -t detr/detections -v
```

---

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

---

## 👨‍💻 Author

**Kaiwei @ Inventec**  
Software Engineer | Edge AI & Computer Vision

---

## 📝 License

This project is licensed under the **MIT License**.
