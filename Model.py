from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


#Initializing Hands
mp_hands = mp.solutions.hands#responsoble for detecting hands and 21 landmarks
hands = mp_hands.Hands(
    static_image_mode=True, #tells mediapipe that the input is a single image, not a video stream
    max_num_hands=2,# maximum number of hands detected
    min_detection_confidence=0.25, #Detection must be 50% confident to be accepted. If the detected object has below 50% confidence score, it will ignore it
)

mp_drawing = mp.solutions.drawing_utils # utility function for drawing landmarks etc.

valid_file = {".jpg", ".jpeg", ".png"}

x = [] # features (the x,y,z landmarks)
y = [] # labels (a,b,c,d ...)

#Pre processing - adding keypoints to ASL images
path = Path("C://Users//zylge//Downloads//asl_dataset") # get the path of the dataset.


#loops through all files and subfolders insided tha path and returns the full path of each items
for items in path.iterdir():
    image_folder_path = items

    #check if items is a directory
    if not items.is_dir():
        print("Is not a directory")
        continue

    #loops through every image or subfolder of the current subfolder and returns the full path of each items
    for image_items in image_folder_path.iterdir():

        #skip non-image file
        if image_items.suffix.lower() not in valid_file:
            print("non-image file")
            continue

        image_paths = str(image_items)

        read_image = cv2.imread(image_paths)#load image

        #skip unreadable images
        if read_image is None:
            print("unreadable image")
            continue

        image_rgb = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)# Convert BGR to RGB format. Open cv reads images as BGR format. Mediapipe expects RGB format

        #process image
        #results; multi_hand_landmarks ->list of hands with 21 landmarks
        #resutls: multi_handedness -> info about left/right hand
        results = hands.process(image_rgb)# runs the hand detection  and landmarks


        #Check if at least one hand exists
        if results.multi_hand_landmarks: # returns one or two lists containing 21 landmarks depending on how many hands detected
            for hands_idx in results.multi_hand_landmarks:
                #list of compiled landmarks of the hand
                compiled_features = []
                # loops the landmarks
                for index,hand_landmark in enumerate(hands_idx.landmark):#hand_landmark is a tuple or list of x,y,z
                    h,w,_ = read_image.shape
                    #since x,y,z are normalized (between 0 -1) multiply it with width and height respectively
                    compiled_features.extend([hand_landmark.x,hand_landmark.y,hand_landmark.z])
                    #print(hand_landmark.x * w, hand_landmark.y * h, hand_landmark.z)

                #Check the length of compiled_feature list(should be 63 -> 21*3)
                if len(compiled_features) == 63:
                    x.append(compiled_features)
                    y.append(image_folder_path.name)
                if len(compiled_features) != 63:
                    print("compiled feature != 63")

                #Optional for diagram: draw landmarks on images.
                #mp_drawing.draw_landmarks(read_image,hands_idx,mp_hands.HAND_CONNECTIONS)

        else:
            print("Did not detect hand "+str(image_items))


        # Optional for diagram: Show result
        # cv2.imshow("Hand Landmarks", read_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

print(len(x))
print(len(y))


#Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Train model
random_classifier_model = RandomForestClassifier(n_estimators=100, max_depth=None,random_state=42)
random_classifier_model.fit(x_train, y_train)

#make prediction
y_pred = random_classifier_model.predict(x_test)

#Model Evaluation

print("Evaluation")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))












