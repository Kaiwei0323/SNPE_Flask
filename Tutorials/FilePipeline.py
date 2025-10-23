import gi
import queue as Q
from gi.repository import Gst, GstApp, GLib
import numpy as np
import cv2
import time
import os
# os.environ["GST_PLUGIN_FEATURE_RANK"] = "v4l2h264dec:0"

class FilePipeline:
    def __init__(self, uri, image_queue, capture_lock):
        self.uri = uri.replace("file://", "") if uri.startswith("file://") else uri
        self.pipeline = None  # Will hold the pipeline reference
        self.bus = None
        self.loop = None
        self.image_queue = image_queue  # Queue to store image frames
        self.capture_lock = capture_lock
        
        self.filesrc = Gst.ElementFactory.make("filesrc", "filesrc")
        self.qtdemux = Gst.ElementFactory.make("qtdemux", "qtdemux")
        self.queue_demux = Gst.ElementFactory.make("queue", "queue_demux")
        self.h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
        self.decoder = Gst.ElementFactory.make("v4l2h264dec", "decoder")  # Hardware
        
        self.queue = Gst.ElementFactory.make("queue", "queue")
        self.capsfilter_nv12 = Gst.ElementFactory.make("capsfilter", "capsfilter_nv12")
        self.videoconvert = Gst.ElementFactory.make("qtivtransform", "qtivtransform")
        
        self.videoscale = Gst.ElementFactory.make("videoscale", "videoscale")
        self.capsfilter_rgb = Gst.ElementFactory.make("capsfilter", "capsfilter_rgb")
        self.videorate = Gst.ElementFactory.make("videorate", "videorate")
        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        
        self.rate = 1

        # Check if elements were created successfully
        if not all([self.filesrc, self.qtdemux, self.queue_demux, self.h264parse,
            self.decoder, self.queue, self.capsfilter_nv12, self.videoconvert, self.videoscale, self.capsfilter_rgb, self.videorate, self.appsink]):
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
        if self.pipeline:
            self.pipeline.set_state(Gst.State.READY)  # Prepare the pipeline for restart
            time.sleep(1)
            self.pipeline.set_state(Gst.State.PLAYING)         
            
    def create(self):
    
        # Set properties
        self.filesrc.set_property("location", self.uri)
        self.h264parse.set_property("disable-passthrough", True)
        self.h264parse.set_property("config-interval", 1)

        self.capsfilter_nv12.set_property("caps", Gst.Caps.from_string("video/x-raw,format=NV12"))
        
        self.videoconvert.set_property("engine", "fcv")
        
        self.capsfilter_rgb.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB,width=1280,height=720"))
        
        # Set the framerate property for the videorate element
        self.videorate.set_property("rate", self.rate) 

        # Configure appsink properties
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.connect("new-sample", self.on_new_sample)

        # Create the pipeline
        self.pipeline = Gst.Pipeline.new(self.uri)
        

        # Add elements to the pipeline
        elements = [
            self.filesrc, self.qtdemux, self.queue_demux, self.h264parse,
            self.decoder, self.queue, self.capsfilter_nv12, self.videoconvert,
            self.capsfilter_rgb, self.videoscale, self.videorate, self.appsink
        ]
        
        # Link the elements together
        for element in elements:
            self.pipeline.add(element)

        # Connect dynamic pad to the queue
        self.qtdemux.connect("pad-added", self.on_pad_added, self.queue)  
        
        # Static links
        self.filesrc.link(self.qtdemux)
        self.queue_demux.link(self.h264parse)
        self.h264parse.link(self.decoder)
        self.decoder.link(self.queue)
        self.queue.link(self.capsfilter_nv12)
        self.capsfilter_nv12.link(self.videoconvert)
        self.videoconvert.link(self.capsfilter_rgb)
        self.capsfilter_rgb.link(self.videoscale)
        self.videoscale.link(self.videorate)
        self.videorate.link(self.appsink)

        # Setup bus
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_message)

        print("Elements linked successfully")

    def start(self):
        # Start playing the pipeline
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

    def on_pad_added(self, demux, pad, queue):
        # Get the pad's capabilities (caps)
        caps = pad.query_caps(None)
        structure = caps.get_structure(0)
        media_type = structure.get_name()

        # Only link video pads
        if media_type.startswith("video"):
            sink_pad = self.queue_demux.get_static_pad("sink")
            if not sink_pad.is_linked():
                ret = pad.link(sink_pad)
                if ret == Gst.PadLinkReturn.OK:
                    print("qtdemux pad linked to queue_demux")
                else:
                    print("Failed to link qtdemux pad")
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
