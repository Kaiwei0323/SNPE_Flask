import gi
import queue as Q
from gi.repository import Gst, GstApp, GLib
import numpy as np
import cv2
import time
import os
from .BasePipeline import BasePipeline

class WebcamPipeline(BasePipeline):
    def __init__(self, uri, image_queue, capture_lock):
        super().__init__(uri, image_queue, capture_lock)
        
        self.v4l2src = Gst.ElementFactory.make("v4l2src", "v4l2src")
        self.capsfilter_h264 = Gst.ElementFactory.make("capsfilter", "capsfilter_h264")
        self.h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
        self.decoder = Gst.ElementFactory.make("v4l2h264dec", "decoder")

        # Check if elements were created successfully
        if not all([self.v4l2src, self.capsfilter_h264, self.h264parse, self.decoder]):
            print("Not all elements could be created")
            return

        print("Created all elements successfully")      
            
    def create(self):
        # Set properties
        self.v4l2src.set_property("device", self.uri)
        
        self.h264parse.set_property("disable-passthrough", True)
        self.h264parse.set_property("config-interval", 1)
        
        self.capsfilter_h264.set_property("caps", Gst.Caps.from_string("video/x-h264, format=NV12"))         
        
        # Add elements to the pipeline
        elements = [
            self.v4l2src, self.capsfilter_h264, self.h264parse, self.decoder, self.queue, self.capsfilter_nv12, self.videoconvert,
            self.capsfilter_rgb, self.videoscale, self.videorate, self.appsink
        ]
        
        # Link the elements together
        for element in elements:
            self.pipeline.add(element)
        
        # Static links
        self.v4l2src.link(self.capsfilter_h264)
        self.capsfilter_h264.link(self.h264parse)
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
