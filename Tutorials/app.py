from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import os
from importlib import import_module
import paho.mqtt.client as mqtt
import requests
import json
import subprocess
import signal

# Import the camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

CAMERA_SOURCES = {}

app = Flask(__name__)

@app.route('/')
def home():
    """Home page with navigation buttons."""
    return render_template('home.html')

@app.route('/vision_solution')
def vision_solution():
    """Video streaming home page."""
    return render_template('index.html', camera_sources=CAMERA_SOURCES)

@app.route('/add_camera', methods=['POST'])
def add_camera():
    """Add a new camera source from form submission."""
    camera_name = request.form['camera_name']
    video_source = request.form['video_source']
    model = request.form['model']
    runtime = request.form['runtime']
    
    if video_source == "RTSP":
        video_source = request.form['rtsp_url']
    
    # Update CAMERA_SOURCES with new camera information
    CAMERA_SOURCES[camera_name] = {
        "source": video_source,
        "model": model,
        "runtime": runtime,
        "camera_instance": Camera(video_source, model, runtime)
    }

    return redirect(url_for('vision_solution'))

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

    return redirect(url_for('vision_solution'))

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
    
@app.route('/delete_all_cameras', methods=['POST'])
def delete_all_cameras():
    """Delete all cameras."""
    camera_names = list(CAMERA_SOURCES.keys())  # Create a list of camera names to iterate over
    
    for camera_name in camera_names:
        camera_instance = CAMERA_SOURCES[camera_name].get("camera_instance")
        if camera_instance:
            camera_instance.stop()  # Stop the camera and cleanup
            print(f"Stopped camera: {camera_name}")
        del CAMERA_SOURCES[camera_name]  # Remove camera from the sources
    
    print("All cameras have been deleted.")
    return redirect(url_for('home'))  # Redirect back to home


@app.route('/smart_farm_demo', methods=["GET", "POST"])
def smart_farm_demo():
    # Initialize variables
    farm_name = sensor_ip = port = ""  # Ensure all variables are initialized
    sensor_data = {}

    if request.method == "POST":
        # Get form data
        farm_name = request.form["farm_name"]
        sensor_ip = request.form["sensor_ip"]
        port = request.form["port"]

        # Create URLs for the four sensor endpoints
        endpoints = {
            "light": f"http://{sensor_ip}:{port}/light",
            "temperature": f"http://{sensor_ip}:{port}/temperature",
            "air": f"http://{sensor_ip}:{port}/air",
            "humidity": f"http://{sensor_ip}:{port}/humidity"
        }

        def fetch_sensor_data(sensor, url):
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    return {"error": f"Failed to fetch data (Status Code: {response.status_code})"}

                data = response.json()
                mqtt_message = data.get('mqtt_message', '{}')
                if mqtt_message == '{}':
                    return {"error": "Empty mqtt_message"}

                sensor_json = json.loads(mqtt_message)
                return extract_sensor_value(sensor, sensor_json)

            except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
                return {"error": f"Error fetching or parsing data: {e}"}

        def extract_sensor_value(sensor, sensor_json):
            # Map each sensor to its corresponding value in the JSON
            mapping = {
                'light': 'brightness',
                'temperature': 'temperature',
                'air': 'pollution_rate',
                'humidity': 'humidity'
            }
            return {sensor: sensor_json.get(mapping.get(sensor), 'N/A')}

        # Process each sensor's data
        for sensor, url in endpoints.items():
            sensor_data.update(fetch_sensor_data(sensor, url))

    # Ensure to pass port even if it hasn't been set yet (in case of a GET request)
    return render_template("smart_farm.html", farm_name=farm_name, sensor_ip=sensor_ip, port=port, sensor_data=sensor_data)
    
# To store the port-forward process for each session (since we don't have persistent session management here)
active_processes = {}

def get_k8s_services():
    command = ["kubectl", "get", "svc", "-n", "deviceshifu", "-o", "json"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        return None, result.stderr.decode("utf-8")
    
    services = json.loads(result.stdout)
    
    formatted_services = []
    for item in services.get("items", []):
        name = item["metadata"]["name"]
        svc_type = item["spec"]["type"]
        cluster_ip = item["spec"]["clusterIP"]
        external_ip = item["status"].get("loadBalancer", {}).get("ingress", [{}])[0].get("ip", "<pending>")
        port = item["spec"]["ports"][0]["port"] if item["spec"]["ports"] else None
        age = item["metadata"].get("creationTimestamp", "Unknown")
        
        formatted_services.append({
            "name": name,
            "type": svc_type,
            "cluster_ip": cluster_ip,
            "external_ip": external_ip,
            "port": port,
            "age": age
        })
    
    return formatted_services, None

# Function to forward port asynchronously and return the PID
def port_forward(service_name, ip_address, port):
    command = [
        "kubectl", "port-forward", "-n", "deviceshifu", 
        f"svc/{service_name}", f"{port}:{80}", "--address", ip_address
    ]
    
    # Start the process asynchronously
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Store the process ID to kill it later
    pid = process.pid
    active_processes[pid] = process

    return pid, None

# Function to stop the port forwarding by killing the process
def stop_port_forward(pid):
    process = active_processes.get(pid)
    
    if process:
        # Kill the process using its PID
        process.kill()
        del active_processes[pid]
        return True, None
    else:
        return False, "Process not found."

@app.route('/my_kubernetes')
def my_kubernetes():
    return render_template('my_kubernetes.html')

@app.route('/get_services', methods=['GET'])
def get_services():
    services, error = get_k8s_services()
    if error:
        return jsonify({"status": "error", "message": error}), 500
    
    return jsonify({"status": "success", "services": services})

@app.route('/port_forward', methods=['POST'])
def handle_port_forward():
    service_name = request.form.get('service_name')
    ip_address = request.form.get('ip_address')
    port = request.form.get('port')

    if not service_name or not ip_address or not port:
        return jsonify({"status": "error", "message": "All fields are required!"}), 400

    pid, error = port_forward(service_name, ip_address, port)
    
    if error:
        return jsonify({"status": "error", "message": error}), 500
    
    return jsonify({"status": "success", "message": f"Port forwarding started on {ip_address}:{port}", "pid": pid})

@app.route('/stop_port_forward', methods=['POST'])
def handle_stop_port_forward():
    pid = request.form.get('pid')

    if not pid:
        return jsonify({"status": "error", "message": "PID is required!"}), 400
    
    try:
        pid = int(pid)
        success, error = stop_port_forward(pid)
        
        if success:
            return jsonify({"status": "success", "message": "Port forwarding stopped successfully!"})
        else:
            return jsonify({"status": "error", "message": error}), 500
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid PID!"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)

