import numpy as np
import cv2
import mtcnn
import random
import time

cap = cv2.VideoCapture(0)
detector = mtcnn.MTCNN()

while(True):
    count = 0 
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    while True:
        name = "Bryan"
        # 1. Import image
        _, img = cap.read()
        cv2.imshow("Image", img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
          break
    cap.release()
    cv2.destroyAllWindows()

    