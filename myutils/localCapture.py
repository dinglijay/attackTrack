# run this program on each RPi to send a labelled image stream
import cv2
import socket
import time
import imagezmq
import numpy as np

from threading import Thread
# from imutils.video import VideoStream

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream", resolution="480p"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)

        # set resolution
        resolution = STD_DIMENSIONS[resolution]
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # read
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
		# keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class VideoStream:
    def __init__(self, src=0, resolution=(320, 240), **kwargs):
        self.stream = WebcamVideoStream(src=src, resolution='720p')

    def start(self):
        # start the threaded video stream
        return self.stream.start()

    def update(self):
        # grab the next frame from the stream
        self.stream.update()

    def read(self):
        # return the current frame
        return self.stream.read()

    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()


if __name__ == "__main__":

    resolution = '720p'
    sender = imagezmq.ImageSender(connect_to='tcp://137.82.223.118:6025')

    rpi_name = socket.gethostname() # send RPi hostname with each image
    picam = VideoStream(src=0, resolution=resolution).start()
    time.sleep(2.0)  # allow camera sensor to warm up
    while True: 
        image = picam.read()
        # image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) 
        image = np.transpose(image, (1,0,2))
        sender.send_image(rpi_name, image)