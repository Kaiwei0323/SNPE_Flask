from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from importlib import import_module
import paho.mqtt.client as mqtt
import sys

from camera import model_map

# Import the camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

CAMERA_SOURCES = {}

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    user = os.environ.get("USER", "default_user") 
    model_options = [model for model in model_map.keys()]
    return render_template('index.html', user=user, camera_sources=CAMERA_SOURCES, model_options=model_options)

@app.route('/add_camera', methods=['POST'])
def add_camera():
    """Add a new camera source from form submission."""
    camera_name = request.form['camera_name']
    video_source = request.form['video_source']
    model = request.form['model']
    runtime = request.form['runtime']
    
    print(f"Adding camera: {camera_name}, Video Source: {video_source}, Model: {model}, Runtime: {runtime}")
    
    if video_source == "RTSP":
        video_source = request.form['rtsp_url']
        print(f"RTSP URL: {video_source}")
    elif video_source == "/dev/video":
        video_source = "/dev/video" + request.form['webcam_idx']
        print(f"Webcam index: {video_source}")

    # Update CAMERA_SOURCES with new camera information
    CAMERA_SOURCES[camera_name] = {
        "source": video_source,
        "model": model,
        "runtime": runtime,
        "camera_instance": Camera(video_source, model, runtime)
    }

    return redirect(url_for('index'))


@app.route('/delete_camera', methods=['POST'])
def delete_camera():
    """Delete a camera source."""
    camera_name = request.form['camera_name']
    if camera_name in CAMERA_SOURCES:
        camera_instance = CAMERA_SOURCES[camera_name].get("camera_instance")
        if camera_instance:
            camera_instance.stop()  # Stop the camera and cleanup
            print(f"Stopped camera: {camera_name}")
        del CAMERA_SOURCES[camera_name]  # Remove camera from the sources

    return redirect(url_for('index'))

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

    # Retrieve the existing camera instance
    camera_instance = video_source["camera_instance"]
    
    return Response(
        gen(camera_instance),  # Use the existing camera instance
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    # Default values
    host = "0.0.0.0"
    port = 5001
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if "--host=" in arg:
                host = arg.split("=")[1]
            if "--port=" in arg:
                port = int(arg.split("=")[1])

    app.run(host=host, port=port, debug=True)

