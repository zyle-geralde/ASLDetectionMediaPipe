import cv2
import mediapipe as mp
import numpy as np

#Hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) # Open default camera

#Check if camera is Opened
if not cap.isOpened():
    print("Camera not opened")
    exit()

while True:
    #ret -> returns a boolean if frame is successfuly cpatured
    ret,frame = cap.read()#capture from camera

    if not ret:
        print("Frame not captured")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result_frame = hands.process(frame_rgb)

    if result_frame.multi_hand_landmarks:
        for hands_idx in result_frame.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hands_idx, mp_hands.HAND_CONNECTIONS)

    #Show frame
    cv2.imshow("Webcam",frame)

    #Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release camera and close window
cap.release()
cv2.destroyAllWindows()
