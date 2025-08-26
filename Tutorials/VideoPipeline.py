import gi
import queue as Q
from gi.repository import Gst, GstApp, GLib
import numpy as np
import cv2
import time
import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "v4l2h264dec:0"

class VideoPipeline:
    def __init__(self, uri, image_queue, capture_lock):
        self.uri = uri  # Set URI for the video stream
        self.pipeline = None  # Will hold the pipeline reference
        self.bus = None
        self.loop = None
        self.image_queue = image_queue  # Queue to store image frames
        self.capture_lock = capture_lock
        
        # Create GStreamer elements and assign them to instance variables
        self.uridecodebin = Gst.ElementFactory.make("uridecodebin", "uridecodebin")
        self.uridecodebin.set_property("use-buffering", True)
        self.queue = Gst.ElementFactory.make("queue", "queue")
        #self.videoconvert = Gst.ElementFactory.make("qtivtransform", "qtivtransform")
        
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
        self.videoscale = Gst.ElementFactory.make("videoscale", "videoscale")
        self.capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        self.videorate = Gst.ElementFactory.make("videorate", "videorate")
        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        
        self.rate = 1

        # Check if elements were created successfully
        if not all([self.uridecodebin, self.queue, self.videoconvert, self.videoscale, self.capsfilter, self.videorate, self.appsink]):
            print("Not all elements could be created")
            return

        print("Created all elements successfully")
        
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
            # Pause if buffering < 100% and resume when ready
            if percent < 100:
                self.pipeline.set_state(Gst.State.PAUSED)
            else:
                self.pipeline.set_state(Gst.State.PLAYING)

    def reconnect(self):
        print("Reconnecting pipeline...")
        self.pipeline.set_state(Gst.State.NULL)  # Stop the pipeline
        self.pipeline.set_state(Gst.State.READY)  # Prepare the pipeline for restart
        self.pipeline.set_state(Gst.State.PLAYING)        
            
    def create(self):
        # Set the URI property of uridecodebin
        self.uridecodebin.set_property("uri", self.uri)

        # Create the caps for the desired video format (e.g., 640x480, RGB format)
        caps = Gst.Caps.from_string("video/x-raw,format=RGB,width=1080,height=580")
        self.capsfilter.set_property("caps", caps)
        
        # Set the framerate property for the videorate element
        self.videorate.set_property("rate", self.rate) 

        # Configure appsink properties
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.connect("new-sample", self.on_new_sample)

        # Create the pipeline
        self.pipeline = Gst.Pipeline.new(self.uri)

        # Add elements to the pipeline
        self.pipeline.add(self.uridecodebin)
        self.pipeline.add(self.queue)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.videoscale)
        self.pipeline.add(self.capsfilter)
        self.pipeline.add(self.videorate)
        self.pipeline.add(self.appsink)

        # Link the elements together
        self.uridecodebin.connect("pad-added", self.on_pad_added, self.queue)  # Connect dynamic pad to the queue
        self.queue.link(self.videoconvert)
        self.videoconvert.link(self.videoscale)
        self.videoscale.link(self.capsfilter)
        self.capsfilter.link(self.videorate)
        self.videorate.link(self.appsink)

        # Connect to the EOS signal to detect end of stream
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_message)

        print("Elements linked successfully")

    def start(self):
        # Start playing the pipeline
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.loop = GLib.MainLoop()
            self.loop.run()

    def destroy(self):
        # Clean up
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            print("Pipeline set to NULL (stopped)")

    def on_pad_added(self, uridecodebin, pad, queue):
        # Get the pad's capabilities (caps)
        caps = pad.query_caps(None)
        structure = caps.get_structure(0)
        media_type = structure.get_name()

        # Only link video pads
        if media_type.startswith('video'):
            if not pad.is_linked():
                link_result = pad.link(queue.get_static_pad("sink"))
                if link_result == Gst.PadLinkReturn.OK:
                    print(f"Pad added and linked successfully for {media_type}")
                else:
                    print(f"Failed to link pad: {link_result}")
            else:
                print(f"Pad already linked for {media_type}")
        else:
            print(f"Skipping non-video pad: {media_type}")

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
