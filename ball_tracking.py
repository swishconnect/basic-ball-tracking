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
    frame = imutils.resize(frame, width=600)
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

    #find contours in the mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    #intialize the current (x, y) center of the ball
    center = None

    if len(contours) > 0: #this process occurs when at least when contour is found

        '''
        so you find the largest contour in the mask here.
        then you use that largest contour to find the minimum enclosing circle and centroid
        '''
        largestContour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largestContour)
        moments = cv2.moments(largestContour)
        center = ( int(moments["m10"]) / moments["m00"], int(moments["m01"]), moments["m00"]) #(x, y) using the moments of the largest contour

        #this only happens if the radius is a minimum size
        if radius > 10:
            #draw the circle and the centroid on the frame and update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), radius=int(radius), color=(0, 255, 255), thickness=2) #i think the 2 was supposed to be thickness but i could have been wrong, they didnt do the radius= thing like i did
            cv2.circle(frame, center, 5, (0, 0, 255), -1) #idk what this is

    points.appendleft(center) #update points queue

    for index in range(1, len(points)): #looping through the points

        #if either of the tracked points is None then ignore them 
        if points[index - 1] is None or points[index] is None:
            continue #this will contniue the loop from the start using the next value

        #if there is a point then find the thickness of the line and draw the connecting lines
        thickness = int(numpy.sqrt(args["buffer"] / float(index + 1)) * 2.5)
        cv2.line(img=frame, point1=points[index - 1], point2=points[index], color=(0, 0, 255), thickness=thickness)
        cv2.line()

    #show frame to the screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF #what

    #when 'q' key pressed, stop the loop
    if key == ord('q'):
        break

if not args.get("video", False): #if not using video file then close camera
    vs.stop()
else:
    vs.release() #otherwise realease camera? 

cv2.destroyAllWindows() #close up shop