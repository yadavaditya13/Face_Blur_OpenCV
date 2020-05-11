# importing required packages

from imutils.video import VideoStream
import numpy as np 
import argparse
import imutils
import time
import cv2
import os

# parsing arguments

ap = argparse.ArgumentParser()

ap.add_argument("-f", "--face", required=True, type=str, help="path to face detector model!")
ap.add_argument("-m", "--confidence", type=float, default=0.5, help="minimum confidence value required for filtering weak detections!")

args = vars(ap.parse_args())

# loading face detector model from disk

print("[INFO] Loading Face_Detector_Model from disk...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
print("\n[INFO] Successfully Loaded Face_Detector_Model from disk...")

# initializing the model
print("[INFO] Initializing Face-Detector-Model...")
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# initiating videoStream and giving 2 secs to the camera to warmup
print("[INFO] We are going Live...")
vs = VideoStream(src=0).start()
time.sleep(2)

# let's read frames one by one
while True:
    # grabbing frames
    frame = vs.read()

    # resizing the frame and storing height and width
    frame = imutils.resize(frame, width=700, height=650)
    (h, w) = frame.shape[:2]

    # lets initiate blobbing in the image
    frame_x = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_x, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # lets pass the obtained blobs to face-detector for face-detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        # grabbing confidence value of each face
        confidence = detections[0, 0, i, 2]

        # filtering weak detections
        if confidence > args["confidence"]:

            # grabbing the face box dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # let's make sure that box does not lies outside the frame/image
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # let's extract the face from image
            face = frame[startY:endY, startX:endX]

            # passing the face to gaussianBlur
            blur_face = cv2.GaussianBlur(face, (55, 55), 0)

            # overlaying the blur_face onto our image
            frame[startY:endY, startX:endX] = blur_face

    # let's display the blurred face image

    cv2.imshow("Blurred Face LiveCam: ", frame)
    key = cv2.waitKey(1) & 0xFF

    # if user presser "q" then end the program
    if key == ord("q"):
        break

# cleanUp
vs.stop()
cv2.destroyAllWindows()