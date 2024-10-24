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
CAMERA_SOURCES = {
    # 'webcam1': {"source": "/dev/video0", "runtime": 30, "model": "DETR"},  # First webcam
    # Uncomment the next line for a second webcam
    # 'webcam2': {"source": "/dev/video2", "runtime": 30, "model": "DETR"},
    # Uncomment the next line for an RTSP stream
    # 'rtsp1': {"source": "rtsp://192.168.8.243:8554/mystream", "runtime": 30, "model": "DETR"},
    # 'video1-DSP': {"source": "test_video/traffic.mp4", "model": "YOLOV8", "runtime": "DSP"},
    'video1-DSP': {"source": "test_video/freeway.mp4", "model": "YOLOV8_DSP", "runtime": "DSP"},
    # 'video2-DSP': {"source": "test_video/fall.mp4", "model": "YOLOV8_FALL_DSP", "runtime": "DSP"},
    #'video2-GPU': {"source": "test_video/fall.mp4", "model": "YOLOV8_GPU", "runtime": "GPU"},
    # 'video2-GPU': {"source": "rtsp://99.64.152.69:8554/mystream4", "model": "YOLOV8", "runtime": "GPU"},
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

