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
from snpehelper_manager import PerfProfile, Runtime, SnpeContext
import time
from detr_coco80_class import DETR_COCO80_CLASSES
from detr_fall_class import DETR_FALL_CLASSES
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

class DETR(SnpeContext):
    def __init__(self, dlc_path: str = "None", 
                 input_layers: list = [], 
                 output_layers: list = [], 
                 output_tensors: list = [], 
                 runtime: str = Runtime.CPU, 
                 classes: list = DETR_COCO80_CLASSES,
                 profile_level: str = PerfProfile.BURST, 
                 enable_cache: bool = False):
        super().__init__(dlc_path, input_layers, output_layers, output_tensors, runtime, profile_level, enable_cache)
        self.classes = classes

    def preprocess(self, frame):
        """Preprocess the input frame for the DETR model."""
        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.Resize(800),  # Resize image
            T.ToTensor(),   # Convert to tensor
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
        ])
        img = transform(image).unsqueeze(0)
        out = torch.nn.functional.interpolate(img, size=(480, 480), mode='bicubic', align_corners=False)
        input_image = out.numpy().transpose(0, 2, 3, 1).astype(np.float32)[0].flatten()
        self.SetInputBuffer(input_image, self.m_input_layers[0])

    def postprocess(self, frame, inference_start_time):
        """Process the model's output and update the frame with detected objects."""
        # Get model outputs
        prob = self.GetOutputBuffer(self.m_output_tensors[0]).reshape(1, 100, len(self.classes))
        boxes = self.GetOutputBuffer(self.m_output_tensors[1]).reshape(1, 100, 4)

        # Convert outputs to tensors
        tensor_prob = torch.from_numpy(prob)
        tensor_boxes = torch.from_numpy(boxes)
        
        frame_h, frame_w = frame.shape[:2]
        probas = tensor_prob.softmax(-1)[0, :, :-1]
        
        # Keep only the boxes with high confidence
        keep = probas.max(-1).values > 0.9
        bboxes_scaled = self.rescale_bboxes(tensor_boxes[0, keep], (frame_h, frame_w))

        if keep.sum() == 0:
            print("No boxes kept after thresholding.")
            return frame  # Return original frame if no boxes are kept

        # Draw bounding boxes and labels on the frame
        for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled.tolist()):
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cl = p.argmax()
            text = f'{self.classes[cl]}: {p[cl]:0.2f}'
            cv2.putText(frame, text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            obj = ObjectData(xmin, ymin, xmax, ymax, self.classes[cl], p[cl])
            obj.inference_time = time.time() - inference_start_time
            # Publish detection result via MQTT
            self.publish_detection(obj)

        return frame  # Return the modified frame

    def publish_detection(self, obj):
        label_index = self.classes.index(obj.label) if obj.label in self.classes else -1
    
        # If the label is not found, you can either handle it differently or return early
        if label_index == -1:
            print(f"Error: label '{obj.label}' not found in classes.")
            return
        """Publish detection results to an MQTT topic in JSON format."""
        detection_info = {
            'label': self.classes[label_index],  # Use the label index to fetch the class name
            'confidence': float(obj.conf),  # Convert tensor to float
            'bbox': {key: float(value) for key, value in obj.bbox.items()},  # Convert bbox values to float
            'inference_time': obj.inference_time
        }
    
        # Convert the detection_info dictionary to a JSON string
        detection_json = json.dumps(detection_info)
    
        # Publish the JSON string to the MQTT topic
        mqtt_client.publish("detr/detections", detection_json)

    def box_cxcywh_to_xyxy(self, x):
        """Convert bounding boxes from center-width-height format to xyxy format."""
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, frame_size):
        """Rescale bounding boxes to match original frame size."""
        frame_h, frame_w = frame_size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([frame_w, frame_h, frame_w, frame_h], dtype=torch.float32)
        return b
        
    def initialize(self):
        print("Initializing model...")
        try:
            success = self.Initialize()
            if not success:
                print("Initialization failed!")
            else:
                print("Model initialized successfully.")
        except Exception as e:
            print(f"Initialization Error: {e}")  
        
    def inference(self, frame):
        """Run inference on the frame and return the processed frame."""
        start_time = time.time()
        self.preprocess(frame)
        self.execute()
        frame = self.postprocess(frame, start_time)
       
        return frame

    def execute(self):
        """Execute the model and handle errors."""
        try:
            if not self.Execute():
                print("Model execution failed!")
        except Exception as e:
            print(f"An error occurred during model execution: {e}")
            
