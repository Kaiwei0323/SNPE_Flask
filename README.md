# 🚀 SNPE Flask Setup Guide

> Vision solution using SNPE SDK for real-time inference, Kubernetes-wrapped sensors for smart farming, and Wav2Vec2 for voice recognition.

![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2020.04-blue?logo=ubuntu)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![SNPE](https://img.shields.io/badge/SNPE-v2.26.0.240828-red?logo=qualcomm)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🧠 Supported Models
- ✅ YOLOv8  
- ✅ YOLOv11  
- ✅ DETR_Resnet101  

---

## 🖥️ Hardware Requirements

| Component | Specification |
|----------|----------------|
| Platform | **QCS6490** |
| CPU      | Octa-Core Kryo 670 |
| GPU      | Qualcomm Adreno 643 |

---

## ⚙️ Software Requirements

- OS: Ubuntu 20.04 (arm64)
- SNPE SDK Version: **v2.26.0.240828**
- Python 3.10

---

## 📁 Directory Structure

```bash
Documents/
├── SNPE_Flask/
│   └── v2.26.0.240828/
Videos/
├── freeway.mp4
├── ppe.mp4
├── fall.mp4
├── brain_tumor.mp4
└── med_ppe.mp4
```

---

## 🛠️ Setup Steps

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
source ~/.bashrc
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

## 8. MQTT Setup (Optional)

### Enable Mosquitto Service
```bash
systemctl enable mosquitto
systemctl status mosquitto
```

### Subscribe to Topics

#### 🕒 Detection Time
```bash
mosquitto_sub -h localhost -t detection_time -v
```

#### 📦 YOLOv8 Detection
```bash
mosquitto_sub -h localhost -t yolov8/detections -v
```

#### 🧠 DETR Detection
```bash
mosquitto_sub -h localhost -t detr/detections -v
```

---

## 🧰 Troubleshooting

- ❌ **Mosquitto not starting?**  
  Ensure the service is enabled and properly installed.  
  Try:  
  ```bash
  systemctl restart mosquitto
  ```

- ❌ **Python dependency issues?**  
  Try:  
  ```bash
  python3.10 -m pip install -r requirements.txt
  ```

---

## 👨‍💻 Author

**Kaiwei @ Inventec**  
Software Engineer | Edge AI & Computer Vision

---

## 📝 License

This project is licensed under the **MIT License**.
