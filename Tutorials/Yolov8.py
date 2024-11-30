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
from coco80_class import COCO80_CLASSES
from fall_class import FALL_CLASSES
from ppe_class import PPE_CLASSES
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
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.Resize((640, 640)),  # Resize to 640x640
            T.ToTensor(),          # Convert to tensor
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Normalize
        ])
        img = transform(image).unsqueeze(0).numpy().transpose(0, 2, 3, 1).astype(np.float32).flatten()
        self.SetInputBuffer(img, self.m_input_layers[0])

    def calcIoU(self, ObjectDataA, ObjectDataB):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        xA = max(ObjectDataA.bbox['x'], ObjectDataB.bbox['x'])
        yA = max(ObjectDataA.bbox['y'], ObjectDataB.bbox['y'])
        xB = min(ObjectDataA.bbox['x'] + ObjectDataA.bbox['width'], ObjectDataB.bbox['x'] + ObjectDataB.bbox['width'])
        yB = min(ObjectDataA.bbox['y'] + ObjectDataA.bbox['height'], ObjectDataB.bbox['y'] + ObjectDataB.bbox['height'])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = ObjectDataA.bbox['width'] * ObjectDataA.bbox['height']
        boxBArea = ObjectDataB.bbox['width'] * ObjectDataB.bbox['height']
        
        return interArea / float(boxAArea + boxBArea - interArea) if boxAArea + boxBArea > 0 else 0

    def nms(self, object_data_list, nmsThresh):
        """Apply Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes."""
        if not object_data_list:
            return []
        
        object_data_list.sort(key=lambda obj: obj.conf, reverse=True)
        suppressed = [False] * len(object_data_list)

        for i in range(len(object_data_list)):
            if suppressed[i]:
                continue
            for j in range(i + 1, len(object_data_list)):
                if self.calcIoU(object_data_list[i], object_data_list[j]) > nmsThresh:
                    suppressed[j] = True

        return [obj for i, obj in enumerate(object_data_list) if not suppressed[i]]

    def postprocess(self, frame, inference_start_time):
        """Post-process the model output and draw bounding boxes."""
        if frame is None or frame.size == 0:
            print("Received an empty frame for postprocessing.")
            return
        output = self.GetOutputBuffer(self.m_output_tensors[0])
        if output is None:
            print("Failed to retrieve output buffer!")
            return frame
        
        output = output.reshape(1, len(self.classes) + 4, 8400)
        tensor_output = torch.from_numpy(output)

        if tensor_output.shape != (1, len(self.classes) + 4, 8400):
            print(f"Unexpected output shape: {tensor_output.shape}")
            return frame

        object_data_list = []  # List to store detected objects
        frame_h, frame_w = frame.shape[:2]

        tensor_boxes = tensor_output[0, :4]  # Bounding boxes
        probas = tensor_output[0, 4:]         # Probabilities

        probas_i, max_idx = probas.max(dim=0)  # Max probabilities and indices
        valid_indices = (probas_i >= 0.5) & (max_idx < len(self.classes))
        
        if valid_indices.sum().item() == 0:
            # print("No valid detections found.")
            return frame

        valid_boxes = tensor_boxes[:, valid_indices].T
        valid_probas = probas_i[valid_indices]
        valid_classes = max_idx[valid_indices]
        
        scaled_boxes = self.rescale_bboxes(valid_boxes, (640, 640), (frame_h, frame_w))
        filtered_boxes = self.nms([ObjectData(*box, cls.item(), conf.item()) for box, cls, conf in zip(scaled_boxes, valid_classes, valid_probas)], 0.45)

        for obj in filtered_boxes:
            x1, y1, width, height = obj.bbox['x'], obj.bbox['y'], obj.bbox['width'], obj.bbox['height']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + width), int(y1 + height)), (0, 255, 0), 2)
            text = f'{self.classes[obj.label]}: {obj.conf:.2f}'
            cv2.putText(frame, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            # print(f"Detected: {self.classes[obj.label]}, Confidence: {obj.conf:.2f}")
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
        detection_json = json.dumps(detection_info)
    
        # Publish the JSON string to the MQTT topic
        mqtt_client.publish("yolov8/detections", detection_json)

    def rescale_bboxes(self, out_bboxes, image_size, frame_size):
        """Rescale bounding boxes to match original frame size."""
        img_w, img_h = image_size
        frame_h, frame_w = frame_size
     
        scale_h = frame_h / img_h
        scale_w = frame_w / img_w

        x_c, y_c, w, h = out_bboxes[:, 0], out_bboxes[:, 1], out_bboxes[:, 2], out_bboxes[:, 3]

        x_c_scaled = ((x_c - w * 0.5) * scale_w)
        y_c_scaled = ((y_c - h * 0.5) * scale_h)
        w_scaled = w * scale_w
        h_scaled = h * scale_h

        return torch.stack([x_c_scaled, y_c_scaled, w_scaled, h_scaled], dim=1)  # Shape (N, 4)
        
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
        
        # print(f"Inference Time: {time.time() - start_time:.4f}s")
        return frame

    def execute(self):
        """Execute the model and handle errors."""
        try:
            if not self.Execute():
                print("Model execution failed!")
        except Exception as e:
            print(f"Execution Error: {e}")

"""
# Initialize MQTT Client
client = mqtt.Client()
client.connect("localhost", 1883, 60)  # Use localhost or your broker's IP
client.loop_start()  # Start the loop to process callbacks

# The rest of the code remains unchanged
if __name__ == "__main__":
    model_object = YOLOV8(
        dlc_path="models/yolov8s_encode_int8.dlc",
        input_layers=["images"],
        output_layers=["/model.22/Concat_5"],
        output_tensors=["output0"],
        runtime=Runtime.DSP,
        classes=COCO80_CLASSES,
        profile_level=PerfProfile.BURST,
        enable_cache=False
    )
    
    # Initialize Model
    model_object.initialize()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference
        processed_frame = model_object.inference(frame)

        if processed_frame is not None:
            # Display the output frame
            cv2.imshow('YOLOv8 Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()  # Stop the MQTT loop
    client.disconnect()  # Disconnect the MQTT client
"""
