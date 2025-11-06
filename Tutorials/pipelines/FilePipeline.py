import gi
import queue as Q
from gi.repository import Gst, GstApp, GLib
import numpy as np
import cv2
import time
import os
from .BasePipeline import BasePipeline

class FilePipeline(BasePipeline):
    def __init__(self, uri, image_queue, capture_lock):
        super().__init__(uri, image_queue, capture_lock)
        
        self.filesrc = Gst.ElementFactory.make("filesrc", "filesrc")
        self.qtdemux = Gst.ElementFactory.make("qtdemux", "qtdemux")
        self.queue_demux = Gst.ElementFactory.make("queue", "queue_demux")
        self.h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
        self.decoder = Gst.ElementFactory.make("v4l2h264dec", "decoder")  # Hardware
        
        self.rate = 1

        # Check if elements were created successfully
        if not all([self.filesrc, self.qtdemux, self.queue_demux, self.h264parse, self.decoder]):
            print("Not all elements could be created")
            return

        print("Created all elements successfully")
            
    def create(self):
        # Set properties
        self.filesrc.set_property("location", self.uri)
        self.h264parse.set_property("disable-passthrough", True)
        self.h264parse.set_property("config-interval", 1)

        # Add elements to the pipeline
        elements = [
            self.filesrc, self.qtdemux, self.queue_demux, self.h264parse,
            self.decoder, self.queue, self.capsfilter_nv12, self.videoconvert,
            self.capsfilter_rgb, self.videoscale, self.videorate, self.appsink
        ]
        
        # Link the elements together
        for element in elements:
            if element is None:
                print(f"❌ Missing element: {element}")
            else:
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
