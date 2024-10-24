import gi
import cv2
import numpy as np
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Create a GStreamer pipeline with appsink
pipeline = Gst.parse_launch("v4l2src device=/dev/video0 ! videoconvert ! video/x-raw, format=rgb ! appsink name=sink")

# Get the appsink element
appsink = pipeline.get_by_name('sink')

# Set appsink properties
appsink.set_property('emit-signals', True)
appsink.set_property('drop', True)
appsink.set_property('max-buffers', 1)

# Define a callback function to handle new samples
def on_new_sample(sink):
    sample = sink.emit('pull-sample')
    buf = sample.get_buffer()
    
    # Extract buffer data
    data = buf.extract_dup(0, buf.get_size())
    
    # Get the caps (format) of the sample
    caps = sample.get_caps()
    structure = caps.get_structure(0)

    # Get width, height, and number of channels
    width = structure.get_value('width')
    height = structure.get_value('height')
    channels = structure.get_value('channels') if structure.has_name('RGB') else 3  # Default to 3 if not available
    
    # Create a NumPy array from the buffer data
    image_array = np.frombuffer(data, np.uint8).reshape((height, width, channels))
    
    # Show the image using OpenCV
    cv2.imshow('Video Stream', image_array)
    
    # Return Gst.FlowReturn.OK indicates that the sample was handled
    return Gst.FlowReturn.OK

# Connect the new-sample signal to the callback function
appsink.connect('new-sample', on_new_sample)

# Start playing the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Create a GLib Main Loop to keep the script running
loop = GLib.MainLoop()

try:
    # Run the main loop
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    # Stop the pipeline and clean up
    pipeline.set_state(Gst.State.NULL)

# Release resources
cv2.destroyAllWindows()
