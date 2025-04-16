import soundfile as sf
import torch
from transformers import Wav2Vec2Processor
import onnxruntime as ort
import numpy as np
import time
import psutil
import pyaudio
import webrtcvad
import threading
from queue import Queue, Empty

class Wave2Vec2ONNXInference:
    def __init__(self, model_name, onnx_path, device_name="default", device_index=0):
        # Load processor for audio feature extraction
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Configure ONNX session options for optimized CPU usage
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        
        # Initialize ONNX model session with specified options and providers
        self.model = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["QNNExecutionProvider"],
            provider_options=[{"backend_path": "./lib/libQnnHtpV68.so"}]
        )
        
        self.device_name = device_name
        self.device_index = device_index
        self.asr_input_queue = Queue(maxsize=120)
        self.asr_output_queue = Queue(maxsize=120)
        self.stop_event = threading.Event()
        self.asr_process = None
        self.vad_process = None
        print("Finished Initializing wav2vec")

    def stop(self):
        """Stop the ASR process gracefully."""
        print("Stopping threads...")
        self.stop_event.set()

        # Send a sentinel to unblock ASR queue
        self.asr_input_queue.put("close")

        if self.vad_process is not None and self.vad_process.is_alive():
            self.vad_process.join(timeout=2)
            self.vad_process = None

        if self.asr_process is not None and self.asr_process.is_alive():
            self.asr_process.join(timeout=2)
            self.asr_process = None

        print("ASR and VAD threads stopped.")

    def start(self):
        """Start the ASR and VAD processes."""
        self.stop_event.clear()
        
        self.asr_process = threading.Thread(
            target=self._asr_process,
            args=(self.model, self.processor, self.asr_input_queue, self.asr_output_queue),
            daemon=True
        )
        self.asr_process.start()

        time.sleep(2)  # Allow ASR model to initialize
        self.vad_process = threading.Thread(
            target=self._vad_process,
            args=(self.device_name, self.asr_input_queue, self.device_index),
            daemon=True
        )
        self.vad_process.start()

    def _vad_process(self, device_name, asr_input_queue, device_index):
        """Voice Activity Detection to capture speech from the microphone."""
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        FRAME_DURATION = 30  # in ms
        CHUNK = int(RATE * FRAME_DURATION / 1000)

        stream = audio.open(input_device_index=device_index,
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''
        try:
            while not self.stop_event.is_set():
                frame = stream.read(CHUNK, exception_on_overflow=False)
                is_speech = vad.is_speech(frame, RATE)

                if is_speech:
                    frames += frame
                else:
                    if len(frames) > 1:
                        if asr_input_queue.full():
                            asr_input_queue.get()
                        asr_input_queue.put(frames)
                    frames = b''

        except Exception as e:
            print(f"VAD Error: {e}")
        finally:
            if len(frames) > 1:
                if asr_input_queue.full():
                    asr_input_queue.get()
                asr_input_queue.put(frames)
            stream.stop_stream()
            stream.close()
            audio.terminate()
            print("VAD thread terminated.")

    def _asr_process(self, model, processor, in_queue, output_queue):
        """ASR process that decodes speech into text."""
        print("\nListening for speech...\n")
        while not self.stop_event.is_set():
            try:
                audio_frames = in_queue.get(timeout=0.5)
                if audio_frames == "close":
                    break

                start_time = time.time()
                audio_np = np.frombuffer(audio_frames, dtype=np.int16)

                inputs = processor(torch.tensor(audio_np), sampling_rate=16000, return_tensors="np", padding=True)
                input_values = inputs.input_values

                onnx_outputs = model.run(None, {model.get_inputs()[0].name: input_values})[0]
                prediction = np.argmax(onnx_outputs, axis=-1)

                transcription = processor.decode(prediction.squeeze().tolist())
                inference_time = time.time() - start_time

                if transcription and transcription != "":
                    if output_queue.full():
                        output_queue.get()
                    output_queue.put([transcription, inference_time])

                # print(f"Inference Time: {inference_time:.4f} seconds, text: {transcription}")

            except Empty:
                continue
            except Exception as e:
                print(f"ASR Error: {e}")
                continue
        print("ASR thread terminated.")

    def get_last_text(self):
        """Retrieve the last transcribed text along with the inference time."""
        print("Get from output queue")
        return self.asr_output_queue.get()


"""
if __name__ == "__main__":
    # Test the ASR system with the pre-trained Wav2Vec2 model
    print("Available ONNX Providers:", ort.get_all_providers())
    print("Starting ASR test...")

    # Initialize the Wave2Vec2 model for ONNX inference
    asr = Wave2Vec2ONNXInference("jonatasgrosman/wav2vec2-large-xlsr-53-english", "wav2vec2-large-xlsr-53-english_quant.onnx", device_name="mic", device_index=0)
    
    # Start the ASR and VAD processes
    asr.start()
    
    try:
        while True:
            # Get the latest transcription and inference time
            text, inference_time = asr.get_last_text()
            print(f"{inference_time:.3f}s\t{text}")

    except KeyboardInterrupt:
        # Stop the ASR process gracefully on keyboard interrupt
        asr.stop()
        print("ASR process terminated.")
"""
