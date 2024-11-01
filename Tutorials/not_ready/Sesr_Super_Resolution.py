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

class SESR(SnpeContext):
    def __init__(self, dlc_path: str = "None",
                 input_layers: list = [],
                 output_layers: list = [],
                 output_tensors: list = [],
                 runtime: str = Runtime.CPU,
                 profile_level: str = PerfProfile.BALANCED,
                 enable_cache: bool = False):
        super().__init__(dlc_path, input_layers, output_layers, output_tensors, runtime, profile_level, enable_cache)

    def preprocess(self, image):
        image_resized = cv2.resize(image, (128, 128))
        preprocess_image = image_resized.flatten() / 255.0
        self.SetInputBuffer(preprocess_image,"image")
        return  

    def postprocess(self):
        OutputBuffer = self.GetOutputBuffer("upscaled_image") 
        OutputBuffer = OutputBuffer.reshape(512,512,3)
        OutputBuffer = OutputBuffer * 255.0
        return OutputBuffer.astype(np.uint8)

if __name__ == "__main__":
    # Initialize the model
    model_object = SESR(
        dlc_path="models/sesr_m5_fp32.dlc",
        input_layers=["image"],
        output_layers=["/model/depth_to_space/DepthToSpace"],
        output_tensors=["upscaled_image"],
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

        start_preprocess_time = time.time()
        model_object.preprocess(frame)
        preprocess_time = time.time() - start_preprocess_time

        # Execute the model
        start_execute_time = time.time()
        if not model_object.Execute():
            print("Model execution failed!")
            break
        execute_time = time.time() - start_execute_time

        # Post-process results and update the frame
        start_postprocess_time = time.time()
        image = model_object.postprocess()
        postprocess_time = time.time() - start_postprocess_time

        # Display timing information
        print(f"Capture Time: {capture_time:.4f}s, Preprocess Time: {preprocess_time:.4f}s, "
              f"Execute Time: {execute_time:.4f}s, Postprocess Time: {postprocess_time:.4f}s")

        # Display the frame with detections
        cv2.imshow("Webcam Inference", frame)
        cv2.imshow("SESR", image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

