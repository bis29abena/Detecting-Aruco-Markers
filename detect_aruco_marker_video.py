# Usage
# python main.py --type "TYPE OF ARUCO DICTIONARY"
# import necessary packages
import imutils
from adjust_contrast import Contrast
from imutils.video import VideoStream
import cv2 as cv
import argparse
import sys
import time

# create an object from the contrast class
image = Contrast()

# construct an argparse to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", default="DICT_ARUCO_ORIGINAL", help="Aruco tag type to detect", type=str)
args = vars(ap.parse_args())

# defines name of each possible Aruco tag opencv supports
ARUCO_DICT = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

# verify that the supplied Aruco tag exists and supported by OpenCv
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] Aruco tag {} is not supported".format(args["type"]))
    sys.exit(0)

# load the Aruco dictionary, grab the Aruco parameters and detect the marker
arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv.aruco.DetectorParameters_create()

# initilize the video stream and allow the camera sensor to warm up
print("[INFO] intializing video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames of the video
while True:
    # grab the frame from the threaded video stream and resize it to have a minimum
    # width of 1000 and call the adjust brightness from the contrast class on each frame
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    frame = image.adjust_brightness(frame)

    (h, w) = frame.shape[:2]

    writer = cv.VideoWriter("test.avi", cv.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))

    # detect aruco markers in the input frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    # verify at least if one Aruco Marker was detected
    if len(corners) > 0:

        # flatten the ARUCO's ids list
        ids = ids.flatten()

        # loop over the detected Aruco markers corners
        for (markerCorner, markerId) in zip(corners, ids):
            # Extract the corners which are always returned as tl, tr, br and bl
            corners = markerCorner.reshape((4, 2))
            (tl, tr, br, bl) = corners

            # convert each returned (x, y) coordinates as integers
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            # Draw the bounding boxes on the aruco markers detected
            cv.line(frame, tl, tr, (0, 255, 0), 2)
            cv.line(frame, tr, br, (0, 255, 0), 2)
            cv.line(frame, br, bl, (0, 255, 0), 2)
            cv.line(frame, bl, tl, (0, 255, 0), 2)

            # computer and draw the center (x, y) coordinates of the aruco marker
            cx = int((tl[0] + br[0]) / 2.0)
            cy = int((tl[1] + br[1]) / 2.0)
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw the Aruco marker Id on the image
            cv.putText(frame, str(markerId), (tl[0], tl[1] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 2)
    writer.write(frame)
    # show the output frame
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF

    # if the q key is pressed break from the loop
    if key == ord("q"):
        break

# when everything is done
# we release the video capture and results write object
# and destroy all windows on the frame
writer.release()
vs.stop()
cv.destroyAllWindows()
