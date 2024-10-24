from flask import Flask, render_template, Response
import cv2
import os
from importlib import import_module
import paho.mqtt.client as mqtt


# Import the camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera_v4l2 import Camera  # Change to your specific camera module if needed

# Set the video sources here
# Model Choice: YOLOV8S_DSP, YOLOV8S_GPU, YOLOV8S_FALL_DSP, YOLOV8L_FALL_DSP, DETR
# Prerecord Video: test_video/fall.mp4, test_video/traffic.mp4, test_video/ppe.mp4
# Webcam: /dev/video0
# RTSP Stream (1 ~ 4): rtsp://99.64.152.69:8554/mystream1
CAMERA_SOURCES = {
    'FALL-YOLOV8S-DSP': {"source": "test_video/fall.mp4", "model": "YOLOV8S_FALL_DSP", "runtime": "DSP"},
    # 'FALL-YOLOV8L-DSP': {"source": "test_video/fall.mp4", "model": "YOLOV8L_FALL_DSP", "runtime": "DSP"},
}

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', camera_sources=CAMERA_SOURCES)

def gen(camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    for frame in camera.frames():  # Iterate over the frames
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

@app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    """Video streaming route for different cameras."""
    video_source = CAMERA_SOURCES.get(camera_name)
    if not video_source:
        return "Camera not found", 404
        
    source = video_source["source"]     
    model = video_source["model"]
    runtime = video_source["runtime"]

    return Response(
        gen(Camera(video_source=source, model=model, runtime=runtime)),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)

