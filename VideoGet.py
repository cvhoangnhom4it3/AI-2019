from threading import Thread
import cv2

class VideoGet:
    """
    Lớp liên tục nhận được các khung từ một đối tượng VideoCapture
    với một chủ đề chuyên dụng.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
    
    def isOpened(self): 
        return self.stream.isOpened()
    
    def release(self):
        return self.stream.release() 
