import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

prevTime = 0
currTime = 0

cap = cv2.VideoCapture(1) 

detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)  
    lmList = detector.findPosition(img)
    img = detector.placeCircle(img, 8)

    if (len(lmList) != 0):
        print(lmList[8])

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2) 

    cv2.imshow("Image", img) 
    cv2.waitKey(1) 