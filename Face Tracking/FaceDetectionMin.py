import cv2
import mediapipe as mp
import time

currTime = 0
prevTime = 0

cap = cv2.VideoCapture("Videos/1.mp4")

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection) 
            boundingBoxFromClass = detection.location_data.relative_bounding_box

            height, width, channels = img.shape

            # Get location of bounding box
            boundingBox = int(boundingBoxFromClass.xmin * width),  \
                  int(boundingBoxFromClass.ymin * height), int(boundingBoxFromClass.width * width),  \
                  int(boundingBoxFromClass.height * height)
            
            cv2.rectangle(img, boundingBox, (0,0,255), 2)
            cv2.putText(img, f'Score: {int(detection.score[0] * 100)}%', (boundingBox[0] - 20, boundingBox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)