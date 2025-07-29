# SNPE Flask App - Docker Setup

## Quick Start

### 1. Build the Docker Image
```bash
cd /home/aim/Documents/SNPE_Flask/Tutorials
docker build -t snpe-flask-app .
```

### 2. Run the Container
```bash
docker run -d --name snpe-flask-app \
  --privileged \
  --device=/dev/video0 \
  --device=/dev/video1 \
  --device=/dev/video2 \
  --device=/dev/video3 \
  --device=/dev/video32 \
  --device=/dev/video33 \
  --device=/dev/adsprpc-smd \
  --device=/dev/adsprpc-smd-secure \
  --device=/dev/subsys_adsp \
  --device=/dev/subsys_cdsp \
  -v /media/aim/dsp/adsp:/media/aim/dsp/adsp:ro \
  -v /home/aim/Documents/SNPE_Flask/logs:/app/logs \
  -v /home/aim/Documents/SNPE_Flask/Tutorials/Videos:/home/aim/Videos \
  -p 5001:5001 \
  -e SNPE_HEXAGON_LIBRARY_PATH=/media/aim/dsp/adsp \
  -e ADSP_LIBRARY_PATH=/media/aim/dsp/adsp \
  -e HEXAGON_ARM_SYSROOT=/media/aim/dsp/adsp \
  snpe-flask-app
```

### 3. Access the Application
Open your browser and go to: `http://localhost:5001`

## Features

✅ **Video Sources Available:**
- Webcam (HD Pro Webcam C920 at `/dev/video2`)
- Video files from `/home/aim/Videos/`
- RTSP streams

✅ **Models Available:**
- YOLOV8S_DSP (CPU fallback)
- DETR
- YOLOV5
- And more...

✅ **Runtime:**
- CPU (DSP falls back to CPU in Docker)

## Management Commands

### Stop the Container
```bash
docker stop snpe-flask-app
```

### Remove the Container
```bash
docker rm snpe-flask-app
```

### View Logs
```bash
docker logs snpe-flask-app
docker logs -f snpe-flask-app  # Follow logs
```

### Access Container Shell
```bash
docker exec -it snpe-flask-app bash
```

### Restart Container
```bash
docker restart snpe-flask-app
```

## Troubleshooting

### Check if Webcam is Working
```bash
docker exec -it snpe-flask-app bash -c "ls -la /dev/video*"
```

### Test Webcam Capture
```bash
docker exec -it snpe-flask-app bash -c "cd /app && python3.10 -c \"import cv2; cap = cv2.VideoCapture('/dev/video2'); print('Webcam opened:', cap.isOpened()); ret, frame = cap.read(); print('Frame captured:', ret); cap.release()\""
```

### Check Available Videos
```bash
docker exec -it snpe-flask-app bash -c "ls -la /home/aim/Videos/"
```

## File Structure

```
Tutorials/
├── app.py                 # Main Flask application
├── camera.py              # Camera handling
├── VideoPipeline.py       # Video processing
├── WebcamPipeline.py      # Webcam handling
├── models/                # AI models
├── templates/             # Web interface
├── myclasses/            # Model classes
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
└── Videos/              # Video files (mounted)
```

## Environment Variables

- `ADSP_LIBRARY_PATH=/media/aim/dsp/adsp`
- `HEXAGON_ARM_SYSROOT=/media/aim/dsp/adsp`
- `SNPE_HEXAGON_LIBRARY_PATH=/media/aim/dsp/adsp`
- `SNPE_DEFAULT_RUNTIME=DSP` (falls back to CPU)

## Ports

- **5001**: Web interface
- **1883**: MQTT (optional)

## Volumes

- `/app/logs` → `/home/aim/Documents/SNPE_Flask/logs`
- `/home/aim/Videos` → `/home/aim/Documents/SNPE_Flask/Tutorials/Videos`
- `/media/aim/dsp/adsp` → `/media/aim/dsp/adsp` (read-only) 