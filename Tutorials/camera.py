import io
import threading
import queue
from PIL import Image
import cv2
import time
# from base_camera import BaseCamera
from snpehelper_manager import PerfProfile, Runtime

from myclasses import *

from VideoPipeline import VideoPipeline
from WebcamPipeline import WebcamPipeline

import gi
from gi.repository import Gst, GstApp, GLib

from mqtt import MQTTClient

import json

mqtt_client = MQTTClient()

model_map = {
    "DETR": ("models/detr_resnet101_int8.dlc", ["image"], ["/model/class_labels_classifier/MatMul_post_reshape", "/model/Sigmoid"], ["logits", "boxes"], DETR_COCO80_CLASSES),
    "DETR_FALL": ("models/fall_detr_int8.dlc", ["pixel_values"], ["/class_labels_classifier/MatMul_post_reshape", "/Sigmoid"], ["logits", "pred_boxes"], DETR_FALL_CLASSES),
    "DETR_PPE": ("models/ppe_detr_int8.dlc", ["pixel_values"], ["/class_labels_classifier/MatMul_post_reshape", "/Sigmoid"], ["logits", "pred_boxes"], DETR_PPE_CLASSES),
    "YOLOV8S_DSP": ("models/yolov8s_encode_int8.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], COCO80_CLASSES),
    "YOLOV8S_GPU": ("models/yolov8s_quantized.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], COCO80_CLASSES),
    "YOLOV8S_FALL_DSP": ("models/yolov8s_fall_encode_int8.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], FALL_CLASSES),
    "YOLOV8L_FALL_DSP": ("models/yolov8l_fall_encode_int8.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], FALL_CLASSES),
    "YOLOV8S_BRAIN_TUMOR_DSP": ("models/yolov8s_brain_tumor_int8.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], BRAIN_TUMOR_CLASSES),
    "YOLOV8S_PPE_DSP": ("models/ppe_int8.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], PPE_CLASSES),
    "YOLOV8S_MED_PPE_DSP": ("models/yolov8s_med_ppe_int8.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], MED_PPE_CLASSES),
}

class Camera():
    """Using OpenCV to capture video frames with threading for inference."""
    def __init__(self, video_source="/dev/video0", model="DETR", runtime="CPU"):
        Gst.init(None)
        
        # Add a parameter to control how often inference happens (e.g., every 5th frame)
        self.infer_every_n_frames = 5
        self.frame_counter = 0  # Initialize the frame counter
        
        self.video_source = video_source
        self.model = model
        self.runtime = self._set_runtime(runtime)
        self.inference_thread = None
        self.capture_thread = None
        self.display_thread = None
        self.info_thread = None
        self.display_lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.inference_frame_queue = queue.Queue(maxsize=60)  # Queue to store frames
        self.capture_frame_queue = queue.Queue(maxsize=60)
        self.model_object = self._initialize_model()
        self.vp = None
        
        self.capture_time = None
        self.inference_time = None
        self.display_time = None
        
        
        if self.video_source.startswith("/dev/video"):
            self.vp = WebcamPipeline(video_source, self.capture_frame_queue, self.capture_lock)
            self.infer_every_n_frames = 5
        else:
            self.vp = VideoPipeline(video_source, self.capture_frame_queue, self.capture_lock)
            if self.video_source.startswith("file://"):
                self.vp.set_rate(0.1)
            self.infer_every_n_frames = 3
        
        self.stop_event = threading.Event()

        if self.model_object is None: 
            raise Exception("Model initialization failed. Exiting.")

        # Start the capture thread
        self.capture_thread = threading.Thread(target=self.start_capture)
        self.capture_thread.start()

        # Start the inference thread
        self.inference_thread = threading.Thread(target=self.start_inference)
        self.inference_thread.start()
        
        self.display_thread = threading.Thread(target=self.frames)
        self.display_thread.start()

        self.info_thread = threading.Thread(target=self.showInfo)
        self.info_thread.start()


    def _set_runtime(self, runtime):
        """Set the runtime based on the specified string."""
        if runtime == "CPU":
            return Runtime.CPU
        elif runtime == "GPU":
            return Runtime.GPU
        elif runtime == "DSP":
            return Runtime.DSP
        else:
            return Runtime.CPU

    def start_capture(self):
        """Initialize the video capture object."""

        # Initialize VideoPipeline here, and call create()
        self.vp.create()  # Ensure VideoPipeline is created
        print("Start Capturing")
        while not self.stop_event.is_set():
            self.vp.start()   # Start video pipeline
        return True  # We don't need the video capture object since we use VideoPipeline

    def _initialize_model(self):
        """Initialize the specified model."""
        try:
            if self.model in model_map:
                dlc_path, input_layers, output_layers, output_tensors, *classes = model_map[self.model]
                return self._load_model(dlc_path, input_layers, output_layers, output_tensors, classes[0] if classes else None)
            else:
                print("Invalid model specified.")
                return None

        except Exception as e:
            print(f"Model initialization error: {e}")
            return None

    def _load_model(self, dlc_path, input_layers, output_layers, output_tensors, classes):
        """Load and initialize the model."""
        from Detr_Object_Detection import DETR
        from Yolov8 import YOLOV8

        if self.model.startswith("DETR"):
            model = DETR(
                dlc_path=dlc_path,
                input_layers=input_layers,
                output_layers=output_layers,
                output_tensors=output_tensors,
                runtime=self.runtime,
                classes=classes,
                profile_level=PerfProfile.BURST,
                enable_cache=False
            )
        else:
            model = YOLOV8(
                dlc_path=dlc_path,
                input_layers=input_layers,
                output_layers=output_layers,
                output_tensors=output_tensors,
                runtime=self.runtime,
                classes=classes,
                profile_level=PerfProfile.BURST,
                enable_cache=False
            )

        model.initialize()
        return model

    def start_inference(self):
        last_frame_time = time.time()
        reconnect_threshold = 5
        
        frame_count = 0
        total_time = 0.0
        """Continuously read frames and perform inference."""
        while not self.stop_event.is_set():  # Check for the stop signal
            curr_time = time.time()
            #ret, img = self.video.read()
            img = None
            with self.capture_lock:
                # print(f"Capture frame queue: {self.capture_frame_queue.qsize()}")
                if not self.capture_frame_queue.empty():
                    img = self.capture_frame_queue.get_nowait()
                    last_frame_time = time.time()
                self.capture_time = time.time() - curr_time
            if img is None:
                elapsed_time = time.time() - last_frame_time
                if self.stop_event.is_set():
                    break
                else:
                    if elapsed_time > reconnect_threshold:
                        print(f"Unable to grab frame for {elapsed_time:.2f} seconds, reconnecting...")
                        # Call the reconnect method here
                        self.vp.reconnect()
                        last_frame_time = time.time()  # Reset the time after reconnect
                    else:
                        # print("Failed to grab a valid frame.")
                        continue
                    
            self.frame_counter += 1  # Increment the frame counter
            if self.frame_counter % self.infer_every_n_frames == 0:  # Check if it's the n-th frame
                with self.display_lock:
                    inference_start = time.time()
                    if self.model_object is not None:
                        processed_frame = self.model_object.inference(img)
                        if not self.inference_frame_queue.full() and processed_frame is not None and processed_frame.size != 0:
                            self.inference_frame_queue.put(processed_frame)
                            # print(f"Display Queue Size: {self.inference_frame_queue.qsize()}")
                        else:
                            print("Dropped Inferenced Frame.")   
                    else:
                        print("Model object is not initialized.")
                    
                    inference_end = time.time()
                    inference_time = inference_end - inference_start

                    # Update total time and frame count for FPS calculation
                    frame_count += 1
                    total_time += inference_time

                    # Calculate and print FPS every second
                    if frame_count >= 30:  # Update FPS every 30 frames (or another interval)
                        fps = frame_count / total_time
                        print(f"FPS: {fps:.2f}")
                        frame_count = 0
                        total_time = 0.0  # Reset for the next period

                    self.inference_time = inference_time	
            else:
                pass
                # print(f"Skipped frame {self.frame_counter} (not every {self.infer_every_n_frames} frame)")
        
        print("Inference loop ended")
                    
    def stop(self):
        """Stop the camera and cleanup."""
        self.vp.destroy()
        self.stop_event.set() 
        with self.display_lock and self.capture_lock:
            if self.inference_thread is not None:
                self.inference_thread.join(timeout=1)  # Wait for the thread to finish
                self.inference_thread = None
            if self.capture_thread is not None:
                self.capture_thread.join(timeout=1)  # Wait for the thread to finish
                self.capture_thread = None
            if self.display_thread is not None:
                self.display_thread.join(timeout=1)  # Wait for the thread to finish
                self.display_thread = None                
            if self.info_thread is not None:
                self.info_thread.join(timeout=1)  # Wait for the thread to finish
                self.info_thread = None             
            
            """
            if self.video is not None:
                self.video.release()  # Release the video capture object
                self.video = None
            """

    def frames(self):
        """Generate frames from the video source with inference."""
        bio = io.BytesIO()

        try:
            while not self.stop_event.is_set():
                with self.display_lock:
                    if not self.inference_frame_queue.empty():
                        display_start_time = time.time()
                        # Get the frame from the queue
                        inference_frame = self.inference_frame_queue.get_nowait()
                        pil_image = Image.fromarray(inference_frame)
                        pil_image.save(bio, format="jpeg")
                        yield bio.getvalue()
                        bio.seek(0)
                        bio.truncate()
                        # print(f"Display Time: {time.time() - display_start_time:.4f}s")
                        
                        self.display_time = time.time() - display_start_time
                
        finally:
            # Ensure the video capture object is released only if it's initialized
            #if self.video is not None:
            #    self.video.release() 
            self.vp.destroy()
            

    def showInfo(self):
        while not self.stop_event.is_set(): 
            # Sleep for a short period to allow other threads to update times
            time.sleep(1)  # Adjust as necessary based on your application

            # Safe handling of None values, assigning 0 if any value is None
            capture_time_ms = self.capture_time * 1000 if self.capture_time is not None else 0
            inference_time_ms = self.inference_time * 1000 if self.inference_time is not None else 0
            display_time_ms = self.display_time * 1000 if self.display_time is not None else 0
            """
            # Print the information
            print("-----------------------------------------------------------")
            print(f"Capture frame queue: | {self.capture_frame_queue.qsize()}   ")
            print(f"Display Queue Size:  | {self.inference_frame_queue.qsize()}")
            print(f"Capture Time:        | {capture_time_ms:.4f}ms")
            print(f"Inference Time:      | {inference_time_ms:.4f}ms")
            print(f"Display Time:        | {display_time_ms:.4f}ms")
            print("-----------------------------------------------------------")
            """
            """Publish detection results to an MQTT topic in JSON format."""
            detection_time_info = {
                'Capture frame queue': self.capture_frame_queue.qsize(),
                'Display Queue Size': self.inference_frame_queue.qsize(),
                'Capturue Time': capture_time_ms,
                'Inference Time': inference_time_ms,
                'Display Time': display_time_ms
            }
    
            # Convert the detection_info dictionary to a JSON string
            detection_time_json = json.dumps(detection_time_info, indent=4)
            # print(detection_time_json)
    
            # Publish the JSON string to the MQTT topic
            mqtt_client.publish("detection_time", detection_time_json)
