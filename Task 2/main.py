import cv2 #pylint: disable=all
import time
import os

def printout(value):
    os.system('cls')
    print(value)

printout('Loading Haar Cascades and variavles')

# Load the Haar cascades XML file
face_cascade = cv2.CascadeClassifier('resources\\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('resources\\haarcascade_eye.xml')

cam_ind = int(input("enter the index of camera you wish to use: "))

# Open the webcam
cap = cv2.VideoCapture(cam_ind)  # 1 corresponds to Droid cam

# Creating variables
t1 = time.time()
frame_count = 0 
fps = None 

printout('Starting the Show press "q" to exit')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # flipping and rotatting the frame as required
    frame = cv2.flip(frame,0)
    frame = cv2.rotate(frame,0)

    # Calculate FPS
    frame_count += 1
    if frame_count % 10 == 0:
        t2 = time.time()
        elapsed_time = t2 - t1
        fps = frame_count / elapsed_time
        t1 = time.time()
        frame_count = 0

    if fps is not None:
        cv2.putText(frame, f'FPS: {round(fps, 0)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0) , 2)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    eyes = eye_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces and label with your name
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle
        cv2.putText(frame, 'Prithvi Srivastava', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)

    # Draw rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle

    # Display the resulting frame
    cv2.imshow('Face & Eye Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

printout('Thanks for watching')
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
