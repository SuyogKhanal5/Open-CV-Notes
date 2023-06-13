import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1) # Start video capture

mpHands = mp.solutions.hands # Create MediaPipe hand object
hands = mpHands.Hands() # Call hand tracking functions
mpDraw = mp.solutions.drawing_utils # Drawing object from MediaPipe

# Variables for tracking frame rate
prevTime = 0
currTime = 0

while True:
    success, img = cap.read() # Read frame from camera

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # We need to convert because hands object only uses RGB images (not sure if this is true)
    results = hands.process(imgRGB) # Process frame in hand tracking and give us the result    

    # print(results.multi_hand_landmarks)

    if (results.multi_hand_landmarks): # multi hand landmarks are the important points on the hand
        for handLms in results.multi_hand_landmarks: # For each hand in the image
            for id, lm in enumerate(handLms.landmark): # Get each landmark and enumerate/index
                # print(id, lm)

                # Get shape of image
                height, width, channels = img.shape

                # Convert graph measurements to pixels
                centerX, centerY = int(lm.x * width), int(lm.y * height)

                print(id, centerX, centerY)

                if id == 8: # ID 8 is the tip of the pointer finger
                    cv2.circle(img, (centerX, centerY), 25, (255, 0, 0), cv2.FILLED) # Image, Position, Radius, Color (BRG), Filled  

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # We want to draw the landmarks on the original image rather than the RGB 

    # Find Frame Rate
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2) # Image, Text, Position, Font, Scale, Color (BGR), Thickness

    cv2.imshow("Image", img) # Show frame on screen
    cv2.waitKey(1) # Wait 