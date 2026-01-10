import cv2
import mediapipe as mp
import numpy as np
import joblib
from NormalizationFunction import normalize_landmarks

#Load model
model = joblib.load("asl_rf_model.pkl")

#Hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
)
prediction_label = "No hand detected"

#Utility for drawing landmark
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
            # Store landmarks
            stored_landmarks = []

            for idx,hand_landmark in enumerate(hands_idx.landmark):
                #Append landmarks
                stored_landmarks.extend([hand_landmark.x,hand_landmark.y, hand_landmark.z])

            #convert into and array
            stored_landmarks = np.array(stored_landmarks)
            if stored_landmarks.shape == (63,):
                # Normalize
                normalized_landmarks = normalize_landmarks(stored_landmarks)
                #Reshape Landmark
                reshaped_array = normalized_landmarks.reshape(1, -1)
                print(reshaped_array.shape)
                #Make prediction
                prediction = model.predict(reshaped_array)
                print("Prediction:",prediction[0])
                prediction_label = prediction[0]

            mp_drawing.draw_landmarks(frame, hands_idx, mp_hands.HAND_CONNECTIONS)
    else:
        prediction_label = "No hand detected"

    cv2.putText(
        frame,
        f"ASL: {prediction_label}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    #Show frame
    cv2.imshow("Webcam",frame)

    #Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release camera and close window
cap.release()
cv2.destroyAllWindows()
