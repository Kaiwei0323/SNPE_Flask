# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from snpe import PerfProfile, Runtime, SnpeContext
import time

from myclasses import *

import paho.mqtt.client as mqtt
from mqtt import MQTTClient
import json

mqtt_client = MQTTClient()

torch.set_grad_enabled(False)

class ObjectData:
    def __init__(self, x, y, width, height, label, conf):
        self.bbox = {'x': x, 'y': y, 'width': width, 'height': height}
        self.label = label
        self.conf = conf   
        self.inference_time = 0.0    

class YOLOV8(SnpeContext):
    def __init__(self, dlc_path: str = "None", 
                 input_layers: list = [], 
                 output_layers: list = [], 
                 output_tensors: list = [], 
                 runtime: str = Runtime.CPU, 
                 classes: list = COCO80_CLASSES,
                 profile_level: str = PerfProfile.BURST, 
                 enable_cache: bool = False):
        super().__init__(dlc_path, input_layers, output_layers, output_tensors, runtime, profile_level, enable_cache)
        self.classes = classes

    def preprocess(self, image):
        """Preprocess the image for YOLOv8 model."""
        if image is None or image.size == 0:
            print("Received an empty frame for preprocessing.")
            return
        
        
        # If image is already the required shape, no need to resize
        if image.shape[:2] != (640, 640):
            image = cv2.resize(image, (640, 640))  # Resize to 640x640 only if necessary
            
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        transform = T.Compose([
            T.ToTensor(),          # Convert to tensor
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Normalize
        ])
        img = transform(image).unsqueeze(0).numpy().transpose(0, 2, 3, 1).astype(np.float32).flatten()
        self.SetInputBuffer(img, self.m_input_layers[0])

    def calcIoU(self, ObjectDataA, ObjectDataB):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
    
        # Early return if either bounding box has zero area
        boxAArea = ObjectDataA.bbox['width'] * ObjectDataA.bbox['height']
        boxBArea = ObjectDataB.bbox['width'] * ObjectDataB.bbox['height']
    
        if boxAArea == 0 or boxBArea == 0:
            return 0

        # Compute the coordinates of the intersection box
        xA = max(ObjectDataA.bbox['x'], ObjectDataB.bbox['x'])
        yA = max(ObjectDataA.bbox['y'], ObjectDataB.bbox['y'])
        xB = min(ObjectDataA.bbox['x'] + ObjectDataA.bbox['width'], ObjectDataB.bbox['x'] + ObjectDataB.bbox['width'])
        yB = min(ObjectDataA.bbox['y'] + ObjectDataA.bbox['height'], ObjectDataB.bbox['y'] + ObjectDataB.bbox['height'])
    
        # Compute intersection area
        interArea = max(0, xB - xA) * max(0, yB - yA)
    
        # Compute the union area using the precomputed box areas
        unionArea = boxAArea + boxBArea - interArea
    
        # Return the IoU
        return interArea / unionArea if unionArea > 0 else 0


    def nms(self, object_data_list, nmsThresh):
        """Apply Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes."""
        if not object_data_list:
            return []
    
        # Sort the list of bounding boxes by confidence score in descending order
        object_data_list.sort(key=lambda obj: obj.conf, reverse=True)

        # List to store the final bounding boxes
        selected_boxes = []

        # Iterate through the sorted list of object data
        for i in range(len(object_data_list)):
            if object_data_list[i] is None:  # Skip suppressed boxes
                continue

            # Add the current box to the selected list
            selected_boxes.append(object_data_list[i])

            # Suppress boxes that have a high IoU with the current box
            for j in range(i + 1, len(object_data_list)):
                if object_data_list[j] is not None and self.calcIoU(object_data_list[i], object_data_list[j]) > nmsThresh:
                    object_data_list[j] = None  # Mark the box as suppressed

        return selected_boxes

    def postprocess(self, frame, inference_start_time):
        """Post-process the model output and draw bounding boxes."""
        if frame is None:
            print("Received an empty frame for preprocessing.")
            return frame
        
        # Retrieve output buffer from the model
        output = self.GetOutputBuffer(self.m_output_tensors[0])
        if output is None:
            print("Failed to retrieve output buffer!")
            return frame
        
        # Reshape the output
        output = output.reshape(1, len(self.classes) + 4, 8400)
        tensor_output = torch.from_numpy(output)

        if tensor_output.shape != (1, len(self.classes) + 4, 8400):
            print(f"Unexpected output shape: {tensor_output.shape}")
            return frame

        object_data_list = []  # List to store detected objects
        frame_h, frame_w = frame.shape[:2]

        tensor_boxes = tensor_output[0, :4]  # Bounding boxes
        probas = tensor_output[0, 4:]         # Probabilities

        # Check if probas is a PyTorch tensor or NumPy array
        if isinstance(probas, torch.Tensor):
            # For PyTorch tensors, use 'dim' argument
            probas_i, max_idx = probas.max(dim=0)  # Max probabilities and indices
        else:
            # For NumPy arrays, use 'axis' argument
            probas_i = probas.max(axis=0)[0]  # Max probabilities
            max_idx = probas.argmax(axis=0)  # Indices of max probabilities
        
        valid_indices = (probas_i >= 0.5) & (max_idx < len(self.classes))
        
        if valid_indices.sum().item() == 0:
            return frame

        valid_boxes = tensor_boxes[:, valid_indices].T
        valid_probas = probas_i[valid_indices]
        valid_classes = max_idx[valid_indices]
        
        # Rescale bounding boxes
        scaled_boxes = self.rescale_bboxes(valid_boxes, (640, 640), (frame_h, frame_w))
        
        # Apply NMS (Non-Maximum Suppression)
        filtered_boxes = self.nms(
            [ObjectData(*box, cls.item(), conf.item()) for box, cls, conf in zip(scaled_boxes, valid_classes, valid_probas)],
            0.45
        )

        # Draw bounding boxes on the frame
        for obj in filtered_boxes:
            x1, y1, width, height = obj.bbox['x'], obj.bbox['y'], obj.bbox['width'], obj.bbox['height']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + width), int(y1 + height)), (0, 255, 0), 2)
            text = f'{self.classes[obj.label]}: {obj.conf:.2f}'
            cv2.putText(frame, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            obj.inference_time = time.time() - inference_start_time

            # Publish detection result via MQTT
            self.publish_detection(obj)

        return frame

        
    def publish_detection(self, obj):
        """Publish detection results to an MQTT topic in JSON format."""
        detection_info = {
            'label': self.classes[obj.label],
            'confidence': float(obj.conf),  # Convert tensor to float
            'bbox': {key: float(value) for key, value in obj.bbox.items()},  # Convert bbox values to float
            'inference_time': obj.inference_time
        }
    
        # Convert the detection_info dictionary to a JSON string
        detection_json = json.dumps(detection_info, indent=4)
    
        # Publish the JSON string to the MQTT topic
        mqtt_client.publish("yolov8/detections", detection_json)

    def rescale_bboxes(self, out_bboxes, image_size, frame_size):
        """Rescale bounding boxes to match original frame size."""
        # Unpack image and frame sizes
        img_w, img_h = image_size
        frame_h, frame_w = frame_size

        # Calculate scale factors for width and height
        scale_w, scale_h = frame_w / img_w, frame_h / img_h

        # Extract the bounding box components
        x_c, y_c, w, h = out_bboxes[:, 0], out_bboxes[:, 1], out_bboxes[:, 2], out_bboxes[:, 3]

        # Vectorized calculations for scaled bounding box components
        x_c_scaled = (x_c - 0.5 * w) * scale_w
        y_c_scaled = (y_c - 0.5 * h) * scale_h
        w_scaled = w * scale_w
        h_scaled = h * scale_h

        # Return the rescaled bounding boxes as a tensor with shape (N, 4)
        return torch.stack([x_c_scaled, y_c_scaled, w_scaled, h_scaled], dim=1)
        
    def initialize(self):
        """Initialize the model."""
        print("Initializing model...")
        try:
            success = self.Initialize()  # Assuming Initialize is a method in the base class or library
            if not success:
                print("Initialization failed!")
            else:
                print("Model initialized successfully.")
        except Exception as e:
            print(f"Initialization Error: {e}")

    def inference(self, frame):
        """Run inference on the frame and return the processed frame."""
        try:
            start_time = time.time()
            self.preprocess(frame)  # Assuming preprocess is defined
            
            execute_start_time = time.time()
            self.execute()  # Assuming execute is defined
            
            postprocess_start_time = time.time()
            frame = self.postprocess(frame, start_time)  # Assuming postprocess is defined
            
            return frame
        except Exception as e:
            print(f"Inference Error: {e}")
            return frame  # Return original frame in case of an error

    def execute(self):
        """Execute the model."""
        try:
            if not self.Execute():  # Assuming Execute is a method from the base class or library
                print("Model execution failed!")
        except Exception as e:
            print(f"Execution Error: {e}")
