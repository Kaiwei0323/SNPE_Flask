from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import os
from importlib import import_module
import paho.mqtt.client as mqtt
import requests
import json
import subprocess
import signal

import numpy as np
import pickle

import sys

REC_MODEL = pickle.load(open('naive_bayes_model.pkl', 'rb'))
FERT_MODEL = pickle.load(open('random_forest_model.pkl', 'rb'))

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
    return render_template('vision.html', camera_sources=CAMERA_SOURCES)

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

@app.route('/update_inference_frequency', methods=['POST'])
def update_inference_frequency():
    try:
        # Parse the incoming JSON request
        data = request.get_json()

        # Check if 'camera_name' and 'infer_every_n_frames' are in the request body
        if not data or 'infer_every_n_frames' not in data or 'camera_name' not in data:
            return jsonify({"error": "Both 'camera_name' and 'infer_every_n_frames' are required"}), 400

        # Extract the camera name and new frequency value
        camera_name = data['camera_name']
        new_frequency = data['infer_every_n_frames']
        
        # Ensure frequency is greater than zero
        if new_frequency <= 0:
            return jsonify({"error": "Frequency must be greater than 0"}), 400

        # Retrieve the camera instance from CAMERA_SOURCES
        camera_data = CAMERA_SOURCES.get(camera_name)
        if not camera_data:
            return jsonify({"error": f"Camera '{camera_name}' not found"}), 404

        camera_instance = camera_data["camera_instance"]
        
        # Update the Camera object's frequency
        camera_instance.infer_every_n_frames = new_frequency

        # Return a success message with the updated frequency
        response = {
            "message": f"Inference frequency for '{camera_name}' updated to {new_frequency}",
            "current_frequency": camera_instance.infer_every_n_frames
        }

        return jsonify(response), 200

    except Exception as e:
        # Log the exception for better debugging
        print(f"Error updating inference frequency: {e}")
        return jsonify({"error": str(e)}), 500


    
@app.route('/croprecommendation/<res1>/<res2>')
def cropresult(res1, res2):
    # Convert the sensor values from the URL (e.g., JSON-encoded list) or pass as part of the response
    # Here, we assume the sensor values are passed as a query string in the URL, or can be derived otherwise.
    sensor_values = request.args.getlist('sensor_values')

    print(res1)
    corrected_result1 = res1
    print(res2)
    corrected_result2 = res2

    # Pass the results and sensor values to the template
    return render_template('croprecresult.html', corrected_result1=corrected_result1, corrected_result2=corrected_result2, sensor_values=sensor_values)


@app.route('/croprecommendation', methods=['GET', 'POST'])
def cr():
    if request.method == 'POST':
        # Get the user's input for sensor IP, port, and protocol
        sensor_ip = request.form.get('sensor_ip')
        sensor_port = request.form.get('sensor_port')

        # Validate the input (simple check)
        if not sensor_ip or not sensor_port:
            return "Invalid input! Please provide sensor IP and port."

        # Define the list of sensor names
        sensor_names = ['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Initialize an empty list to hold the sensor values
        sensor_values = []

        # Fetch data for each sensor using the IP and port from user input
        for sensor in sensor_names:
            try:
                # Construct the API URL dynamically using the user input
                url = f'http://{sensor_ip}:{sensor_port}/{sensor}'
                
                # Make the GET request to fetch the sensor value
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception if the request fails

                # Assuming the response is a JSON object containing the sensor value in 'mqtt_message'
                response_json = response.json()
                
                if isinstance(response_json, int):
                    sensor_value = response_json
                else:

                    # Extract the actual sensor value from the 'mqtt_message' field
                    sensor_data = response_json.get("mqtt_message", "{}")
                    sensor_json = json.loads(sensor_data)  # Parse the inner JSON
                    sensor_value = sensor_json.get(sensor, 0)  # Default to 0 if the value is missing

                sensor_values.append(float(sensor_value))  # Assuming the value is numeric

            except requests.exceptions.RequestException as e:
                # Handle the error and append a default value if there's an issue fetching the sensor data
                print(f"Error fetching {sensor}: {e}")
                sensor_values.append(0)  # Default value in case of error

        # Crop prediction - use all 7 sensor values for crop prediction
        crop_input = np.array(sensor_values)  # Use all 7 features for crop prediction
        crop_input = crop_input.reshape(1, -1)

        # Crop prediction using the REC_MODEL
        res1 = REC_MODEL.predict(crop_input)[0]  # Crop prediction result

        # Fertilizer prediction - use only the first 3 features (nitrogen, phosphorous, potassium)
        fertilizer_input = np.array(sensor_values[:3])  # Only take the first 3 values for fertilizer prediction
        fertilizer_input = fertilizer_input.reshape(1, -1)

        # Fertilizer prediction using the FERT_MODEL
        res2 = FERT_MODEL.predict(fertilizer_input)[0]  # Fertilizer prediction result

        # Crop and Fertilizer dictionaries
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        fertilizer_dict = {
            0: 'Urea',
            1: 'DAP',
            2: 'Fourteen-Thirty Five-Fourteen',
            3: 'Twenty Eight-Twenty Eight',
            4: 'Seventeen-Seventeen-Seventeen',
            5: 'Twenty-Twenty',
            6: 'Ten-Twenty Six-Twenty Six'
        }

        # Map results to human-readable names
        crop = crop_dict.get(res1, "Unknown Crop")
        fert = fertilizer_dict.get(res2, "Unknown Fertilizer")

        # Redirect to the `cropresult` route and pass results and sensor values as URL parameters
        return redirect(url_for('cropresult', res1=crop, res2=fert, sensor_values=sensor_values))

    return render_template('croprec.html')

    
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
