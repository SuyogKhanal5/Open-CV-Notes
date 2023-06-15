import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

currTime = 0
prevTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for lm in results.pose_landmarks.landmark:
            height, width, channels = img.shape

            x,y = int(lm.x * width), int(lm.y * height)
            cv2.circle(img, (x,y), 10, (255, 0, 0), cv2.FILLED)


    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)

    cv2.waitKey(1)