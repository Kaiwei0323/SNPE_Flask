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
import paho.mqtt.client as mqtt
from mqtt import MQTTClient

mqtt_client = MQTTClient()
torch.set_grad_enabled(False)

class ObjectData:
    def __init__(self, x, y, width, height, label, conf):
        self.bbox = {'x': x, 'y': y, 'width': width, 'height': height}
        self.label = label
        self.conf = conf   
        self.inference_time = 0.0    

class YOLOV5(SnpeContext):
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
        """Preprocess the image for YOLOv5 model."""
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.Resize((640, 640)),  # Resize to 640x640
            T.ToTensor(),          # Convert to tensor
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Normalize
        ])
        img = transform(image).unsqueeze(0).numpy().transpose(0, 2, 3, 1).astype(np.float32).flatten()
        self.SetInputBuffer(img, "images")

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

    def nms(self, boxes, scores, nmsThresh):
        """Apply Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes."""
        if len(boxes) == 0:
            return []

        indices = np.array(range(len(scores)))
        boxes = np.array(boxes)

        # Calculate the area of the boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort the boxes by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        keep = []

        while len(sorted_indices) > 0:
            i = sorted_indices[0]  # index of the box with the highest score
            keep.append(i)

            # Calculate IoU for the remaining boxes
            xx1 = np.maximum(x1[i], x1[sorted_indices[1:]])
            yy1 = np.maximum(y1[i], y1[sorted_indices[1:]])
            xx2 = np.minimum(x2[i], x2[sorted_indices[1:]])
            yy2 = np.minimum(y2[i], y2[sorted_indices[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
 
            intersection = w * h
            iou = intersection / (areas[i] + areas[sorted_indices[1:]] - intersection)

            # Keep boxes with IoU below the threshold
            sorted_indices = sorted_indices[np.where(iou <= nmsThresh)[0] + 1]

        return keep


    def postprocess(self, frame, inference_start_time):
        """Post-process the model output and draw bounding boxes."""
        output = self.GetOutputBuffer("output0")
        print(output)
        if output is None:
            print("Failed to retrieve output buffer!")
            return frame
            
        output = output.reshape(1, 25200, len(self.classes) + 5)
        tensor_output = torch.from_numpy(output)

        if tensor_output.shape != (1, 25200, len(self.classes) + 5):
            print(f"Unexpected output shape: {tensor_output.shape}")
            return frame

        valid_boxes = []  
        scores = []
        frame_h, frame_w = frame.shape[:2]

        for i in range(tensor_output.shape[1]):
            box_data = tensor_output[0][i].view(-1)
            x = box_data[0].item()
            y = box_data[1].item()
            w = box_data[2].item()
            h = box_data[3].item()
            conf_level = box_data[4].item()

            max_idx = torch.argmax(box_data[5:]).item()
            score = box_data[5 + max_idx].item() * conf_level

            if score >= 0.5:
                valid_boxes.append((x, y, w, h))
                scores.append(score)

        if not valid_boxes:
            print("No valid boxes found.")
            return frame

        # Convert valid_boxes to a tensor
        valid_boxes_tensor = torch.tensor(valid_boxes, dtype=torch.float32)

        # Rescale bounding boxes
        scaled_boxes = self.rescale_bboxes(valid_boxes_tensor, (640, 640), (frame_h, frame_w))
        
        # Apply NMS
        keep_indices = self.nms(scaled_boxes.numpy(), scores, nmsThresh=0.45)
        filtered_boxes = [scaled_boxes[i] for i in keep_indices]
        filtered_scores = [scores[i] for i in keep_indices]

        # Create ObjectData instances and draw bounding boxes
        for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
            x1, y1, width, height = box
            obj = ObjectData(x1, y1, width, height, int(max_idx), score)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + width), int(y1 + height)), (0, 255, 0), 2)
            text = f'{self.classes[obj.label]}: {obj.conf:.2f}'
            cv2.putText(frame, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            obj.inference_time = time.time() - inference_start_time
            self.publish_detection(obj)

        return frame
        
    def publish_detection(self, obj):
        """Publish detection results to an MQTT topic."""
        detection_info = {
            'label': self.classes[obj.label],
            'confidence': obj.conf,
            'bbox': obj.bbox,
            'inference_time': obj.inference_time
        }
        mqtt_client.publish("yolov8/detections", str(detection_info))

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
        try:
            self.preprocess(frame)
            self.execute()
            frame = self.postprocess(frame, start_time)
        except Exception as e:
            print(f"Inference Error: {e}")
        return frame

    def execute(self):
        """Execute the model and handle errors."""
        try:
            if not self.Execute():
                print("Model execution failed!")
        except Exception as e:
            print(f"Execution Error: {e}")

# Initialize MQTT Client
client = mqtt.Client()
client.connect("localhost", 1883, 60)  # Use localhost or your broker's IP
client.loop_start()  # Start the loop to process callbacks

if __name__ == "__main__":
    model_object = YOLOV5(
        dlc_path="models/yolov5s_encode_int8.dlc",
        input_layers=["images"],
        output_layers=["/model.24/Concat_3"],
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
            cap = cv2.VideoCapture(0)
            continue

        # Perform inference
        processed_frame = model_object.inference(frame)

        if processed_frame is not None:
            # Display the output frame
            cv2.imshow('YOLOv5 Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()  # Stop the MQTT loop
    client.disconnect()  # Disconnect the MQTT client

