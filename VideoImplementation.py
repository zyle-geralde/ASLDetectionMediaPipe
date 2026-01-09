import cv2

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

    #Show frame
    cv2.imshow("Webcam",frame)

    #Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release camera and close window
cap.release()
cv2.destroyAllWindows()
