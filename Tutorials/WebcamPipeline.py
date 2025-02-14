import gi
import queue as Q
from gi.repository import Gst, GstApp
import numpy as np
import cv2
import time

class WebcamPipeline:
    def __init__(self, uri, image_queue, capture_lock):
        self.uri = uri  # Set URI for the video stream
        self.pipeline = None  # Will hold the pipeline reference
        self.image_queue = image_queue  # Queue to store image frames
        self.capture_lock = capture_lock

        print("Created all elements successfully")

    def create(self):
        self.pipeline = cv2.VideoCapture(self.uri)
        if not self.pipeline.isOpened():
            print(f"Cannot open video source: {self.uri}")
        print(f"Successfully connect webcam: {self.uri}")
        
    def reconnect(self):
        self.pipeline = cv2.VideoCapture(self.uri)
        if not self.pipeline.isOpened():
            print(f"Cannot open video source: {self.uri}")
        print(f"Successfully reconnect webcam: {self.uri}")
    def start(self):
        # Start playing the pipeline
        if self.pipeline.isOpened():
            capture_start_time = time.time()
            ret, img = self.pipeline.read()
            if not ret or img is None:
                print("Fail to grab a valid frame")
            else:
                with self.capture_lock:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.image_queue.qsize() >= 30:
                        drop_frame = self.image_queue.get()
                    self.image_queue.put(img)
            # print(f"Capture Time: {time.time() - capture_start_time:.4f}s")
            
                
    def destroy(self):
        # Clean up and release the video capture object
        if self.pipeline is not None:
            self.pipeline.release()  # Release the webcam
            print("Webcam capture released")
