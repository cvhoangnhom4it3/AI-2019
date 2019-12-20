from threading import Thread
import cv2

class VideoShow:
    """
    Lớp liên tục hiển thị một khung bằng cách sử dụng một luồng chuyên dụng.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
         .stopped = True

    def stop(self):
        self.stopped = True
