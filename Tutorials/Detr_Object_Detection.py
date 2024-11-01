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

torch.set_grad_enabled(False)

class DETR(SnpeContext):
    # COCO classes for object detection
    """
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    """
    CLASSES = [
        'N/A', 'face'
    ]
    
    def __init__(self, dlc_path: str = "None", 
                 input_layers: list = [], 
                 output_layers: list = [], 
                 output_tensors: list = [], 
                 runtime: str = Runtime.CPU, 
                 profile_level: str = PerfProfile.BALANCED, 
                 enable_cache: bool = False):
        super().__init__(dlc_path, input_layers, output_layers, output_tensors, runtime, profile_level, enable_cache)

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
        self.SetInputBuffer(input_image, "input.1")

    def postprocess(self, frame):
        """Process the model's output and update the frame with detected objects."""
        # Get model outputs
        prob = self.GetOutputBuffer("4293").reshape(1, 100, 2)
        boxes = self.GetOutputBuffer("4297").reshape(1, 100, 4)
        
        print(prob)
        print(boxes)

        # Convert outputs to tensors
        tensor_prob = torch.from_numpy(prob)
        tensor_boxes = torch.from_numpy(boxes)
        
        frame_h, frame_w = frame.shape[:2]
        probas = tensor_prob.softmax(-1)[0, :, :-1]
        
        # Keep only the boxes with high confidence
        keep = probas.max(-1).values > 0.3
        bboxes_scaled = self.rescale_bboxes(tensor_boxes[0, keep], (frame_h, frame_w))

        if keep.sum() == 0:
            print("No boxes kept after thresholding.")
            return frame  # Return original frame if no boxes are kept

        # Draw bounding boxes and labels on the frame
        for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled.tolist()):
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cl = p.argmax()
            text = f'{self.CLASSES[cl]}: {p[cl]:0.2f}'
            cv2.putText(frame, text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(f"Detected: {self.CLASSES[cl]}, Confidence: {p[cl]:0.2f}")

        return frame  # Return the modified frame

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
        frame = self.postprocess(frame)
        
        print(f"Inference Time: {time.time() - start_time:.4f}s")
        return frame

    def execute(self):
        """Execute the model and handle errors."""
        try:
            if not self.Execute():
                print("Model execution failed!")
        except Exception as e:
            print(f"An error occurred during model execution: {e}")
            
            
if __name__ == "__main__":
    model_object = DETR(
        dlc_path="models/detr_demo_int8.dlc",
        input_layers=["input.1"],
        output_layers=["/linear_class/MatMul_post_reshape", "/Sigmoid"],
        output_tensors=["4293", "4297"],
        runtime=Runtime.DSP,
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
            cv2.imshow('DETR Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
