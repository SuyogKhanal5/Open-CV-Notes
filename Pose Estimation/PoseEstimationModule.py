import cv2
import mediapipe as mp
import time

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
        
        lmList = []

        if self.results.pose_landmarks:
            for lm in self.results.pose_landmarks.landmark:
                height, width, channels = img.shape

                x,y = int(lm.x * width), int(lm.y * height)

                lmList.append([x,y])

                if draw:
                    cv2.circle(img, (x,y), 10, (255, 0, 0), cv2.FILLED)

        return lmList

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