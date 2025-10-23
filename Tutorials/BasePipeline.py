import gi
import queue as Q
from gi.repository import Gst, GstApp, GLib
import numpy as np
import time

class BasePipeline:
    def __init__(self, uri, image_queue, capture_lock):
        self.uri = uri
        self.pipeline = None
        self.bus = None
        self.loop = None
        self.image_queue = image_queue
        self.capture_lock = capture_lock
        
        self.rate = 1
        
        # Common Elements
        self.queue = Gst.ElementFactory.make("queue", "queue")
        self.capsfilter_nv12 = Gst.ElementFactory.make("capsfilter", "capsfilter_nv12")
        self.capsfilter_nv12.set_property("caps", Gst.Caps.from_string("video/x-raw, format=NV12"))
        self.videoconvert = Gst.ElementFactory.make("qtivtransform", "qtivtransform")
        self.videoconvert.set_property("engine", "fcv")
        self.videoscale = Gst.ElementFactory.make("videoscale", "videoscale")
        self.capsfilter_rgb = Gst.ElementFactory.make("capsfilter", "capsfilter_rgb")
        self.capsfilter_rgb.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB,width=1280,height=720"))
        self.videorate = Gst.ElementFactory.make("videorate", "videorate")
        self.videorate.set_property("rate", self.rate)
        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.connect("new-sample", self.on_new_sample)
        
        self.pipeline = Gst.Pipeline.new(self.uri)
        
        # Check if common elements are created successfully
        if not all([self.queue, self.capsfilter_nv12, self.videoconvert, self.videoscale, self.capsfilter_rgb, self.videorate, self.appsink]):
            print("Not all common elements could be created")
            return
        
        print("Created all common elements successfully")

    def set_rate(self, rate):
        self.rate = rate
        
    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("------------EOS--------------------------")
            self.reconnect()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Warning: {warn}, {debug}")
        elif t == Gst.MessageType.BUFFERING:
            percent = message.parse_buffering()
            print(f"Buffering: {percent}%")
            if percent < 100:
                self.pipeline.set_state(Gst.State.PAUSED)
            else:
                self.pipeline.set_state(Gst.State.PLAYING)

    def reconnect(self):
        print("Reconnecting pipeline...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.READY)
            time.sleep(1)
            self.pipeline.set_state(Gst.State.PLAYING)
        
    def start(self):
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.loop = GLib.MainLoop()
            try:
                self.loop.run()
            except Exception as e:
                print(f"Main loop exited: {e}")
                self.destroy()
            
    def destroy(self):
        # Clean up
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            print("Pipeline set to NULL (stopped)")

    def on_new_sample(self, appsink, data=None):
        # Callback when a new sample (frame) is available from appsink
        #sample = self.appsink.emit("pull-sample")
        sample = self.appsink.pull_sample()
        if isinstance(sample, Gst.Sample):
            buffer = sample.get_buffer()  # Get the buffer from the sample
            caps = sample.get_caps()
            # Extract the width, height, and number of channels
            width = caps.get_structure(0).get_value("width")
            height = caps.get_structure(0).get_value("height")
            channels = 3  # RGB format has 3 channels

            # Extract the buffer data into a numpy array
            buffer_size = buffer.get_size()
            np_array = np.ndarray(shape=(height, width, channels),
                                  dtype=np.uint8,
                                  buffer=buffer.extract_dup(0, buffer_size))

            np_array = np.copy(np_array)
            
            with self.capture_lock:
                # Handle queue overflow by dropping the oldest frame
                if self.image_queue.full():
                    drop_frame = self.image_queue.get()
                    # print("Queue full, dropping oldest frame")

                # Add the new frame to the queue
                self.image_queue.put(np_array)
                # print(f"Frame added to queue. Current queue size: {self.image_queue.qsize()}")

            return Gst.FlowReturn.OK
        else:
            print("Failed to get sample")
            return Gst.FlowReturn.ERROR



