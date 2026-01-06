from pathlib import Path
import mediapipe as mp

#Initializing Hands
mp_hands = mp.solutions.hands#responsoble for detecting hands and 21 landmarks
hands = mp.hands.Hands(
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
        image_paths = image_items








