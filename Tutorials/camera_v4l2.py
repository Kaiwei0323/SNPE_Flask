import io
from PIL import Image
import cv2
from base_camera import BaseCamera
from snpehelper_manager import PerfProfile, Runtime, SnpeContext
from coco80_class import COCO80_CLASSES
from fall_class import FALL_CLASSES

class Camera(BaseCamera):
    """Using OpenCV to capture video frames."""
    def __init__(self, video_source="/dev/video0", model="DETR", runtime="CPU"):
        self.video_source = video_source
        self.model = model
        self.runtime = self._set_runtime(runtime)
        self.video = self._initialize_video_capture()
        
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

    def _initialize_video_capture(self):
        """Initialize the video capture object."""
        video = cv2.VideoCapture(self.video_source)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video.set(cv2.CAP_PROP_FPS, 30)
        return video

    def _reconnect(self):
        """Reconnect to the video source."""
        print("Reconnecting to video source...")
        self.video.release()  # Release the current capture
        self.video = self._initialize_video_capture()  # Reinitialize

    def _initialize_model(self):
        """Initialize the specified model."""
        model_map = {
            "DETR": ("models/detr_resnet101_int8.dlc", ["image"],
                     ["/model/class_labels_classifier/MatMul_post_reshape", "/model/Sigmoid"],
                     ["logits", "boxes"]),
            "YOLOV8S_DSP": ("models/yolov8s_encode_int8.dlc", ["images"],
                           ["/model.22/Concat_5"], ["output0"], COCO80_CLASSES),
            "YOLOV8S_GPU": ("models/yolov8s_quantized.dlc", ["images"],
                           ["/model.22/Concat_5"], ["output0"], COCO80_CLASSES),
            "YOLOV8S_FALL_DSP": ("models/yolov8s_fall_encode_int8.dlc", ["images"],
                                ["/model.22/Concat_5"], ["output0"], FALL_CLASSES),
            "YOLOV8L_FALL_DSP": ("models/yolov8l_fall_encode_int8.dlc", ["images"],
                                ["/model.22/Concat_5"], ["output0"], FALL_CLASSES),
        }
        
        if self.model in model_map:
            dlc_path, input_layers, output_layers, output_tensors, *classes = model_map[self.model]
            return self._load_model(dlc_path, input_layers, output_layers, output_tensors, classes[0] if classes else None)
        else:
            print("Invalid model specified.")
            return None

    def _load_model(self, dlc_path, input_layers, output_layers, output_tensors, classes):
        """Load and initialize the model."""
        from Detr_Object_Detection import DETR
        from Yolov8 import YOLOV8
        
        if self.model == "DETR":
            model = DETR(
                dlc_path=dlc_path,
                input_layers=input_layers,
                output_layers=output_layers,
                output_tensors=output_tensors,
                runtime=self.runtime,
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

    def frames(self):
        """Generate frames from the video source with inference."""
        model_object = self._initialize_model()
        
        bio = io.BytesIO()

        try:
            while True:
                ret, img = self.video.read()
                if not ret or img is None or img.size == 0:
                    print("Failed to grab a valid frame, trying to reconnect.")
                    self._reconnect()
                    continue

                # Model Inference
                try:
                    frame = model_object.inference(img)
                    if frame is None or frame.size == 0:
                        print("Inference returned an invalid frame.")
                        continue
                except Exception as e:
                    print(f"Inference error: {e}")
                    continue

                # Convert frame to RGB
                try:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Color conversion error: {e}")
                    self._reconnect()
                    continue
                
                # Save to BytesIO
                pil_image = Image.fromarray(image)
                pil_image.save(bio, format="jpeg")
                yield bio.getvalue()

                # Reset BytesIO for the next frame
                bio.seek(0)
                bio.truncate()
                
        finally:
            self.video.release()  # Ensure the video capture object is released

