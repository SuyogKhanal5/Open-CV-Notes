import cv2
import mediapipe as mp
import time
import math

class PoseEstimator():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smooth, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.imgRGB)

        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []

        if self.results.pose_landmarks:
            for lm in self.results.pose_landmarks.landmark:
                height, width, channels = img.shape

                x,y = int(lm.x * width), int(lm.y * height)

                self.lmList.append([x,y])

                if draw:
                    cv2.circle(img, (x,y), 10, (255, 0, 0), cv2.FILLED)

        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw = True):
        x1, y1 = self.lmList[p1]
        x2, y2 = self.lmList[p2]
        x3, y3 = self.lmList[p3]

        # Calculate the angle

        angle = math.degrees(math.atan2(y3-y2, x3-x2)-math.atan2(y1-y2,x1-x2))

        if angle < 0:
            angle += 360

        if draw:
            cv2.circle(img, (x1,y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 15, (0, 0, 255), 2)
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.circle(img, (x2,y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (0, 0, 255), 2)
            cv2.line(img, (x2, y2), (x3, y3), (255,255,255), 2)
            cv2.circle(img, (x3,y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3,y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return angle

def main():
    cap = cv2.VideoCapture(0)

    currTime = 0
    prevTime = 0

    detector = PoseEstimator()

    while True:
        success, img = cap.read()

        img = detector.findPose(img)

        lmList = detector.findPosition(img)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Image", img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()