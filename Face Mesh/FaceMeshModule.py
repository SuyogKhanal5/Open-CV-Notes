import cv2
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self, static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5, minTrackCon=0.5, thickness=1, circle_radius=1, color=(0,0,255)):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.minTrackCon = minTrackCon
        
        self.thickness=thickness
        self.circle_radius = circle_radius
        self.color = color

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.static_image_mode, max_num_faces=self.max_num_faces, 
                                                 min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=self.thickness, circle_radius=self.circle_radius, color=self.color)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLandmarks in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                for lm in faceLandmarks.landmark:
                    height, width, channels = img.shape

                    x,y = int(lm.x * width), int(lm.y * height)

        return img
    

def main():
    cap = cv2.VideoCapture("../Videos/3.mp4")

    detector = FaceMeshDetector()

    currTime = 0
    prevTime = 0

    while True:
        success, img = cap.read()

        img = detector.findFaceMesh(img)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()