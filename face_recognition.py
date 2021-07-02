from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import math

#test
def is_mouth_open(landmarks, ar_threshold=0.7):
# Calculate the euclidean distance labelled as A,B,C

    A = math.hypot(landmarks[50][0] - landmarks[58][0], landmarks[50][1] - landmarks[58][1])

    B = math.hypot(landmarks[52][0] - landmarks[56][0], landmarks[52][1] - landmarks[56][1])

    C = math.hypot(landmarks[48][0] - landmarks[54][0], landmarks[48][1] - landmarks[54][1])
    # Calculate the mouth aspect ratio
    # The value of vertical distance A,B is averaged
    mouth_aspect_ratio = (A + B) / (2.0 * C)
    # Return True if the value is greater than the threshold
    if mouth_aspect_ratio > ar_threshold:
        return True, mouth_aspect_ratio
    else:
        return False, mouth_aspect_ratio

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
vs = VideoStream(1).start()
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        condicion, _ = is_mouth_open(shape)
        if condicion:
            print("Boca abierta")

    # show the frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the q key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
