from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green" ball in the HSV color space
# for a basketball, we'll need to change these values
blueLower = (150, 150, 0)
blueUpper = (180, 255, 255)

# initializing list of tracking points by using the argument passed earlier
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0, resolution=(640, 480)).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

while True:
	# grab the current frame; returns a tuple with size 2 IF we're getting it from
    # VideoCapture; otherwise, it's going to actually be a frame
	frame = vs.read()

	# if we're using VideoCapture, then we have to reference the tuple
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame then we have reached the end
	if frame is None:
		break

	# resize the frame to 600px --> makes process time faster and increases FPS
	frame = imutils.resize(frame, width=600)

    # blurring the frame allows us to reduce noise -- focus on ball better
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # convert the frame to HSV color space
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green" -> basically filters out
    # anything that's not in the specified HSV range
	mask = cv2.inRange(hsv, blueLower, blueUpper)

    # series of erosions + dilations removes any remaining blobs
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask -- contours are basically a curve joining all
    # the continuous points that have the same color and intensity
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use it to compute the minimum 
		# enclosing circle and centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)

		# this link explains: https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# update the points queue
	pts.appendleft(center)

    # loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()

