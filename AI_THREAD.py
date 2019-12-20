from datetime import datetime
import numpy as np
import cv2
import math
import pyautogui
import threading
import concurrent.futures
from VideoGet import VideoGet
from VideoShow import VideoShow

class AI_THREAD:
    
    def __init__(self):
        self.counter_defects = 0;
    
    def maskHSV(self, crop_image):
        blur = cv2.GaussianBlur(crop_image, (3,3), 0)
    
        # Thay đổi không gian màu từ BGR -> HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
        # Tạo một hình ảnh nhị phân với màu trắng sẽ là màu da và phần còn lại là màu đen
        mask = cv2.inRange(hsv, np.array([2,0,0]), np.array([20,255,255]))
        return mask
    
    def findContours(self, mask):
        kernel = np.ones((5,5))
    
        # Áp dụng các biến đổi hình thái để lọc nhiễu nền
        dilation = cv2.dilate(mask, kernel, iterations = 1)
        erosion = cv2.erode(dilation, kernel, iterations = 1)    
        
        # Áp dụng Gaussian Blur và Ngưỡng
        filtered = cv2.GaussianBlur(erosion, (3,3), 0)
        ret,thresh = cv2.threshold(filtered, 127, 255, 0)

        # Tìm đường viền (contours)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        return contours, hierarchy

    def findAngle(self, start, end, far):
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
        return angle
    
    def startDetecting(self):
        
        capture = VideoGet(0).start()
        video_shower = VideoShow(capture.frame).start()   

        while capture.isOpened():
            if capture.stopped or video_shower.stopped:
                video_shower.stop()
                capture.stop()
                break
            #Chụp khung hình từ camera
            frame = capture.frame
            
            # Nhận dữ liệu tay từ cửa sổ phụ hình chữ nhật  
            cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
            crop_image = frame[100:300, 100:300]
            
            #1.
            # Áp dụng Gaussian blur
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                exMask = executor.submit(self.maskHSV, crop_image)
                mask = exMask.result()

            #2.
            # Tìm đường viền (contours)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                exContours = executor.submit(self.findContours, mask)
                contours, hierarchy = exContours.result()
            
            #3.
            try:
                contour = max(contours, key = lambda x: cv2.contourArea(x))
                
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
                
                hull = cv2.convexHull(contour)
                
                drawing = np.zeros(crop_image.shape,np.uint8)
                cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
                cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
                
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour,hull)
                
                count_defects = 0
            
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    # 4.
                    #angle = 360;
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        exAngle = executor.submit(self.findAngle, start, end, far)
                        angle = exAngle.result()

                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(crop_image,far,1,[0,0,255],-1)

                    cv2.line(crop_image,start,end,[0,255,0],2)

                if count_defects >= 4:
                    pyautogui.press('space')
                    cv2.putText(frame,"JUMP", (450,110), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

            except:
                pass

            video_shower.frame = frame

            if cv2.waitKey(1) == ord('q'):
                capture.release()
                cv2.destroyAllWindows()
                break 
# def main():
#AI_THREAD.startDetecting(self)

if __name__ == "__main__":
    dino = AI_THREAD()
    dino.startDetecting()
    
