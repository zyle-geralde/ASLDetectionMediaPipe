from pathlib import Path

import cv2
import mediapipe as mp

#Initializing Hands
mp_hands = mp.solutions.hands#responsoble for detecting hands and 21 landmarks
hands = mp_hands.Hands(
    static_image_mode=True, #tells mediapipe that the input is a single image, not a video stream
    max_num_hands=2,# maximum number of hands detected
    min_detection_confidence=0.5, #Detection must be 50% confident to be accepted. If the detected object has below 50% confidence score, it will ignore it
)

mp_drawing = mp.solutions.drawing_utils # utility function for drawing landmarks etc.

#Pre processing - adding keypoints to ASL images
path = Path("C://Users//zylge//Downloads//asl_dataset") # get the path of the dataset.


#loops through all files and subfolders insided tha path and returns the full path of each items
for items in path.iterdir():
    image_folder_path = items

    #loops through every image or subfolder of the current subfolder and returns the full path of each items
    for image_items in image_folder_path.iterdir():
        image_paths = str(image_items)

        read_image = cv2.imread(image_paths)#load image
        image_rgb = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)# Convert BGR to RGB format. Open cv reads images as BGR format. Mediapipe expects RGB format

        #process image
        #results; multi_hand_landmarks ->list of hands with 21 landmarks
        #resutls: multi_handedness -> info about left/right hand
        results = hands.process(image_rgb)# runs the hand detection  and landmarks

        count = 0
        #Check if at least one hand exists
        if results.multi_hand_landmarks: # returns one or two lists containing 21 landmarks depending on how many hands detected
            for hand_landmarks in results.multi_hand_landmarks:
                count+=1
                print(hand_landmarks)

        break
    break








