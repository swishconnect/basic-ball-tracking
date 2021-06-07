from collections import deque #use this to maintain a list of coords of the ball
from imutils.video import VideoStream
import numpy
import argparse
import cv2
import imutils #the blog's personal library for making opencv methods faster
import time

#construct the argument parser
parser = argparse.ArgumentParser()

#parsing the arguments

'''
the video argument gets the argument for the video path
if argument is supplied then it will go to that path for the video
default (when arg not supplied) is using the webcam
'''
parser.add_argument("-v", "--video", help="path to optional video file")

'''
the buffer arg changes the max size of deque
the longer this value, the longer the list of coords for the ball
larger deque = larger contrails
'''
parser.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")

args = vars(parser.parse_args()) #i think this makes a dictionary of all the arguments and puts it in a var called args

#define the lower and upper bounds of the color of the ball, in the tutorial its green (HSV color space)
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

#initialize the list of all tracked points for the ball's coords
points = deque(maxlen=args["buffer"]) #the points are determined by the size of the list (made by deque) which is taken from the buffer arg

#when video path not given, start the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

#when video path is givem, go to it and open it
else:
    vs = cv2.VideoCapture(args["video"])

#wait a little bit for the camera or video to load
time.sleep(2) #waiting 2 seconds


#this loop will keeping going on forever, tracking the ball until something ends its pitiful existence...i mean stops the program
while True:
    #read the current frame
    frame = vs.read()

    #handle the frame
    frame = frame[1] if args.get("video", False) else frame #what
 
    #when we did not get a frame, end the video because it is the end
    if frame is None:
        break #breaks the loop

    #resize the frame, blur it, then convert it to HSV color space
    frame = imutils.resize(width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    '''
    construct a mask for the color "green", then perform
	a series of dilations and erosions to remove any small
	blobs left in the mask
    '''

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    