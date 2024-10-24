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
torch.set_grad_enabled(False)
from PIL import Image
from snpehelper_manager import PerfProfile, Runtime, SnpeContext
import threading
from SafeQueue import SafeQueue
import time

class ObjectData:
    def __init__(self, x, y, width, height, label, conf):
        self.bbox = {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
        self.label = label
        self.conf = conf       

class YOLOX(SnpeContext):
    # COCO classes
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    
    def __init__(self, dlc_path: str = "None",
                 input_layers: list = [],
                 output_layers: list = [],
                 output_tensors: list = [],
                 runtime: str = Runtime.CPU,
                 profile_level: str = PerfProfile.BURST,
                 enable_cache: bool = False):
        super().__init__(dlc_path, input_layers, output_layers, output_tensors, runtime, profile_level, enable_cache)
        

    def preprocess(self, image):
        # Define the preprocessing transformations
        transform = T.Compose([
            T.Resize((640, 640)),  # Resize the image to 640x640 for YOLO
            T.ToTensor(),          # Convert the image to a tensor
            T.Normalize(           # Normalize with YOLO's mean and std values
                mean=[0.0, 0.0, 0.0], 
                std=[1.0, 1.0, 1.0]
            )
        ])
    
        # Apply the transformations
        img = transform(image).unsqueeze(0)  # Add a batch dimension

        # Convert the tensor to a numpy array and flatten it
        input_image = img.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        input_image = input_image[0].flatten()  # Flatten the input image
        
        # Set the input buffer for the model
        self.SetInputBuffer(input_image, "images")
        return
        """
        # Resize the image to 320x320
        img = image.resize((640, 640))

        # Normalize the pixel values to [0, 1] using the scaling factor
        input_scale = 0.00392156862745
        img_array = np.array(img)  # Convert to NumPy array

        # Convert to BGR for OpenCV (if necessary) and apply scaling
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        input_image = img_array.astype(np.float32) * input_scale

        # Flatten the image and set the input buffer for the model
        input_image = input_image.flatten()
        self.SetInputBuffer(input_image, "images")
        return
        """
        
        

    def postprocess(self, image, frame):
        output = self.GetOutputBuffer("output")
        output = output.reshape(1, 8400, 85)
        tensor_output = torch.from_numpy(output)

        object_data_list = []  # List to store detected objects
        
        # Get dimensions of the frame (height, width)
        frame_h, frame_w = frame.shape[:2]  # Shape returns (height, width, channels)
        tensor_output[..., 4:] = torch.sigmoid(tensor_output[..., 4:])  # Corrected line
        # Iterate through each detection
        for i in range(tensor_output.shape[1]):  # 8400 detections
            # Extract boxes and probabilities
            tensor_boxes = tensor_output[0, i, :4]  # Extract the first 4 columns for bounding boxes
            probas = tensor_output[0, i, 5:]           # Remaining columns for class probabilities
            confidence = tensor_output[0, i, 4]  # confidence level
            probas_i = probas.max()                  # Maximum probability
            max_idx = probas.argmax().item()         # Index of the class with the max probability
            print(f"Class: {max_idx}")
            print(f"Max prob: {probas_i:0.5f}")
            print(f"Conf: {confidence:0.9f}")
            if probas_i * confidence >= 0.5 and max_idx < len(self.CLASSES):  # Check index
                bboxes_scaled = self.rescale_bboxes(tensor_boxes, image.size,  (frame_h, frame_w))
                x, y, width, height = bboxes_scaled
                rect = ObjectData(x, y, width, height, max_idx, probas_i)
                object_data_list.append(rect)

        # Process each detected object
        for obj in object_data_list:
            x1 = obj.bbox['x']
            y1 = obj.bbox['y']
            x2 = obj.bbox['x'] + obj.bbox['width']
            y2 = obj.bbox['y'] + obj.bbox['height']

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
            # Safeguard against index errors
            class_label = obj.label
            confidence = obj.conf
        
            text = f'{self.CLASSES[class_label]}: {confidence:0.2f}'
            cv2.putText(frame, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            print(f"Detected: {self.CLASSES[class_label]}, Confidence: {confidence:0.2f}")

        return frame  # Return the modified frame

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x
        b = [(x_c), (y_c),
             (w), (h)]
        return b  # Returning as a list instead of a tensor for easier handling

    def rescale_bboxes(self, out_bbox, image_size, frame_size):
        img_w, img_h = image_size
        frame_h, frame_w = frame_size
        
        # Calculate the scaling factors
        scale_w = frame_h / img_h
        scale_h = (frame_w / img_w) / 1.33


        x_c, y_c, w, h = out_bbox

        # Scale the bounding box coordinates
        x_c_scaled = ((x_c - w * 0.5) * scale_w)
        y_c_scaled = ((y_c - h * 0.5) * scale_h)
        w_scaled = w * scale_w
        h_scaled = h * scale_h
        
        

        # Convert the scaled center coordinates back to (xmin, ymin, xmax, ymax)
        b = self.box_cxcywh_to_xyxy(torch.tensor([x_c_scaled, y_c_scaled, w_scaled, h_scaled]))
        return b  # Returns the bounding box in (xmin, ymin, xmax, ymax) format

if __name__ == "__main__":
    # Initialize the model
    model_object = YOLOX(
        dlc_path="models/yolox_s_int8.dlc",
        input_layers=["images"],
        output_layers=["Transpose_333"],
        output_tensors=["output"],
        runtime=Runtime.GPU,
        profile_level=PerfProfile.BURST,
        enable_cache=False
    )

    print("Initializing model...")
    ret = model_object.Initialize()
    if not ret:
        print("Initialization failed!")
        exit(0)
    print("Model initialized successfully.")

    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        start_capture_time = time.time()
        ret, frame = cap.read()
        capture_time = time.time() - start_capture_time

        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to PIL image for preprocessing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        start_preprocess_time = time.time()
        model_object.preprocess(image)
        preprocess_time = time.time() - start_preprocess_time

        # Execute the model
        start_execute_time = time.time()
        if not model_object.Execute():
            print("Model execution failed!")
            break
        execute_time = time.time() - start_execute_time

        # Post-process results and update the frame
        start_postprocess_time = time.time()
        frame = model_object.postprocess(image, frame)
        postprocess_time = time.time() - start_postprocess_time

        # Display timing information
        print(f"Capture Time: {capture_time:.4f}s, Preprocess Time: {preprocess_time:.4f}s, "
              f"Execute Time: {execute_time:.4f}s, Postprocess Time: {postprocess_time:.4f}s")

        # Display the frame with detections
        cv2.imshow("Webcam Inference", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

