import numpy as np

def normalize_landmarks(landmarks):
    """
    landmarks: np.array of shape (63,) -> 21 points * 3 (x, y, z)
    Returns normalized landmarks centered at wrist and scaled to hand size.
    """
    landmarks = landmarks.reshape(21, 3)

    #Use wrist as origin
    wrist = landmarks[0, :2]  #x, y only
    landmarks[:, :2] -= wrist

    #Scale by hand size
    hand_size = np.linalg.norm(landmarks[9, :2])  #Euclidean distance
    if hand_size > 0:
        landmarks[:, :2] /= hand_size

    #Optionally z-axis normalization if needed
    landmarks[:, 2] -= landmarks[0, 2]  # relative depth

    return landmarks.flatten()