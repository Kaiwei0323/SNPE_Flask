from flask import Flask, render_template, Response
import cv2
from snpehelper_manager import PerfProfile, Runtime, SnpeContext
from Detr_Object_Detection import DETR
from Yolov8 import YOLOV8
from queue import Queue
import threading

app = Flask(__name__)

# Initialize the models
detr_model = DETR(
    dlc_path="models/detr_resnet101_int8.dlc",
    input_layers=["image"],
    output_layers=["/model/class_labels_classifier/MatMul_post_reshape", "/model/Sigmoid"],
    output_tensors=["logits", "boxes"],
    runtime=Runtime.DSP,
    profile_level=PerfProfile.BURST,
    enable_cache=False
)

yolov8_model = YOLOV8(
    dlc_path="models/yolov8s_quantized.dlc",
    input_layers=["images"],
    output_layers=["/model.22/Concat_5"],
    output_tensors=["output0"],
    runtime=Runtime.GPU,
    profile_level=PerfProfile.BURST,
    enable_cache=False
)

# Image queue
frame_queue = Queue(maxsize=10)

def capture_frames(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    while True:
        ret, img = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(img)  # Add frame to queue
        else:
            print("Failed to grab frame.")
            break
    cap.release()

def gen_frame(model, frame_width, frame_height, fps):
    if model == "DETR":
        model_object = detr_model
    elif model == "YOLOV8":
        model_object = yolov8_model
    else:
        print("Invalid model specified.")
        return

    print("Initializing model...")
    ret = model_object.Initialize()
    if not ret:
        print("Initialization failed!")
        exit(0)
    print("Model initialized successfully.")
    
    while True:
        if not frame_queue.empty():
            img = frame_queue.get()  # Get frame from queue
            
            try:
                img = model_object.inference(img)
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Start frame capture in a separate thread
    threading.Thread(target=capture_frames, args=(0,), daemon=True).start()
    return Response(gen_frame("DETR", 640, 480, 30), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

