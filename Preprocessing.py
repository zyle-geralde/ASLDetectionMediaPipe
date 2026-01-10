from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from NormalizationFunction import normalize_landmarks

#PREPROCESSING

#Data augmentation
def augment_data(image):
    augmented_images = []

    h,w,_ = image.shape

    #Original Image
    augmented_images.append(image)

    #Uncomment this only if neede
    # #Horizontal Flip
    # flipped = cv2.flip(image, 1)
    # augmented_images.append(flipped)
    #
    # #I did not include rotation because of I and J which are similar. Inlude this if dataset is small
    # #Rotation
    # for angle in [-15, 15]:
    #     M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    #     rotated = cv2.warpAffine(image, M, (w, h))
    #     augmented_images.append(rotated)
    #
    # #Scale
    # #Excluded "scale up" because scaling up may cause the hand to be partially cropped
    # for scale in [0.9]:
    #     M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    #     scaled = cv2.warpAffine(image, M, (w, h))
    #     augmented_images.append(scaled)
    #
    # #Brightness change
    # brighter = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
    # darker = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
    # augmented_images.append(brighter)
    # augmented_images.append(darker)
    #
    return augmented_images


#Initializing Hands
mp_hands = mp.solutions.hands#responsoble for detecting hands and 21 landmarks
hands = mp_hands.Hands(
    static_image_mode=True, #tells mediapipe that the input is a single image, not a video stream
    max_num_hands=2,# maximum number of hands detected
    min_detection_confidence=0.50, #Detection must be 50% confident to be accepted. If the detected object has below 50% confidence score, it will ignore it
)

mp_drawing = mp.solutions.drawing_utils # utility function for drawing landmarks etc.

valid_file = {".jpg", ".jpeg", ".png"}

x = [] # features (the x,y,z landmarks)
y = [] # labels (a,b,c,d ...)

path = Path("C://Users//zylge//Desktop//Datasets//SignAlphaSet_Sampled") # get the path of the dataset.


#loops through all files and subfolders insided tha path and returns the full path of each items
for items in path.iterdir():
    image_folder_path = items

    #check if items is a directory
    if not items.is_dir():
        print("Is not a directory")
        continue

    # counter for successfully landmarked images in this folder
    success_count = 0
    max_success = 400  #maximum per folder

    #loops through every image or subfolder of the current subfolder and returns the full path of each items
    for image_items in image_folder_path.iterdir():

        # stop if already reached 500 successful images
        if success_count >= max_success:
            print(
                f"Reached {max_success} successfully landmarked images in {image_folder_path.name}, moving to next folder.")
            break

        #skip non-image file
        if image_items.suffix.lower() not in valid_file:
            print("non-image file")
            continue

        image_paths = str(image_items)

        #load image
        original_image = cv2.imread(image_paths)

        if original_image is None:
            continue

        augmented_images = augment_data(original_image)



        for aug_img in augmented_images:
            image_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)# Convert BGR to RGB format. Open cv reads images as BGR format. Mediapipe expects RGB format
            # process image
            # results; multi_hand_landmarks ->list of hands with 21 landmarks
            # resutls: multi_handedness -> info about left/right hand
            results = hands.process(image_rgb)  # runs the hand detection  and landmarks

            # Check if at least one hand exists
            if results.multi_hand_landmarks:  # returns one or two lists containing 21 landmarks depending on how many hands detected
                for hands_idx in results.multi_hand_landmarks:
                    # list of compiled landmarks of the hand
                    compiled_features = []
                    # loops the landmarks
                    for index, hand_landmark in enumerate(
                            hands_idx.landmark):  # hand_landmark is a tuple or list of x,y,z
                        # since x,y,z are normalized (between 0 -1) multiply it with width and height respectively
                        compiled_features.extend([hand_landmark.x, hand_landmark.y, hand_landmark.z])
                        # print(hand_landmark.x * w, hand_landmark.y * h, hand_landmark.z)

                    # Check the length of compiled_feature list(should be 63 -> 21*3)
                    if len(compiled_features) == 63:
                        # Normalize here
                        normalized_features = normalize_landmarks(np.array(compiled_features))
                        x.append(normalized_features)
                        #No normalization
                        #x.append(compiled_features)
                        y.append(image_folder_path.name)
                        success_count += 1  # increment counter
                    if len(compiled_features) != 63:
                        print("compiled feature != 63")

                    # Optional for diagram: draw landmarks on images.
                    #mp_drawing.draw_landmarks(aug_img,hands_idx,mp_hands.HAND_CONNECTIONS)


            else:
                print("Did not detect hand "+str(image_items))


        # Optional for diagram: Show result
        # cv2.imshow("Hand Landmarks", read_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

print(len(x))
print(len(y))

X = np.array(x)
Y = np.array(y)

print("X shape:", X.shape)  # (num_samples, 63)
print("Y shape:", Y.shape)  # (num_samples,)

#Uncomment if you have not saved the file
np.save("asl_landmarks_Xtest.npy", X)
np.save("asl_labels_ytest.npy", Y)

print("Landmarks saved")