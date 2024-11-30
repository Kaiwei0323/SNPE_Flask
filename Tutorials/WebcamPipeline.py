import gi
import queue as Q
from gi.repository import Gst, GstApp
import numpy as np
import cv2

class WebcamPipeline:
    def __init__(self, uri, image_queue):
        self.uri = uri  # Set URI for the video stream
        self.pipeline = None  # Will hold the pipeline reference
        self.image_queue = image_queue  # Queue to store image frames
        
        # Create GStreamer elements and assign them to instance variables
        self.v4l2src = Gst.ElementFactory.make("v4l2src", "v4l2src")
        self.queue = Gst.ElementFactory.make("queue", "queue")
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
        self.videoscale = Gst.ElementFactory.make("videoscale", "videoscale")
        self.capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        self.appsink = Gst.ElementFactory.make("appsink", "appsink")

        # Check if elements were created successfully
        if not all([self.v4l2src, self.queue, self.videoconvert, self.videoscale, self.capsfilter, self.appsink]):
            print("Not all elements could be created")
            return

        print("Created all elements successfully")

    def create(self):
        # Set the URI property of v4l2src
        self.v4l2src.set_property("device", self.uri)

        # Create the caps for the desired video format (e.g., 640x480, RGB format)
        caps = Gst.Caps.from_string("video/x-raw,format=RGB")
        self.capsfilter.set_property("caps", caps)

        # Configure appsink properties
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.connect("new-sample", self.on_new_sample)

        # Create the pipeline
        self.pipeline = Gst.Pipeline.new(self.uri)

        # Add elements to the pipeline
        self.pipeline.add(self.v4l2src)
        self.pipeline.add(self.queue)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.videoscale)
        self.pipeline.add(self.capsfilter)
        self.pipeline.add(self.appsink)

        # Directly link the static "src" pad of v4l2src to the "sink" pad of the queue
        self.v4l2src.link(self.queue)
        self.queue.link(self.videoconvert)
        self.videoconvert.link(self.videoscale)
        self.videoscale.link(self.capsfilter)
        self.capsfilter.link(self.appsink)

        print("Elements linked successfully")

    def start(self):
        # Start playing the pipeline
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.PLAYING)
            """
            print("Pipeline set to PLAYING")
            
            # Set an error handler to capture any errors that might occur
            self.pipeline.get_bus().add_signal_watch()
            self.pipeline.get_bus().connect("message", self.on_bus_message)
            """

    def destroy(self):
        # Clean up
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            print("Pipeline set to NULL (stopped)")

    def on_new_sample(self, appsink, data=None):
        # Callback when a new sample (frame) is available from appsink
        sample = self.appsink.emit("pull-sample")
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

            # Handle queue overflow by dropping the oldest frame
            if self.image_queue.qsize() >= 30:
                drop_frame = self.image_queue.get()
                #print("Queue full, dropping oldest frame")

            # Add the new frame to the queue
            self.image_queue.put(np_array)
            #print(f"Frame added to queue. Current queue size: {self.image_queue.qsize()}")

            return Gst.FlowReturn.OK
        else:
            print("Failed to get sample")
            return Gst.FlowReturn.ERROR

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"Error: {err}, {dbg}")
        elif t == Gst.MessageType.EOS:
            print("End of Stream")

"""
# Function to display frames from the queue
def display_frames(image_queue):
    while True:
        if not image_queue.empty():
            frame = image_queue.get()
            print("Consume frame from queue")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Video Frame", frame_rgb)

        # Check if the user presses 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    Gst.init(None)
    # Initialize the image queue
    image_queue = Q.Queue()
    # Create an instance of the WebcamPipeline class
    
    video_path = "/dev/video0"  # Path to webcam device
    
    vp = WebcamPipeline(video_path, image_queue)

    # Create, start, and destroy the pipeline
    vp.create()  # Initialize and create the pipeline
    vp.start()   # Start playing the pipeline
    display_frames(image_queue)
    vp.destroy() # Clean up and stop the pipeline
"""
