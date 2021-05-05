import numpy as np
import cv2 as openCv

cap = openCv.VideoCapture("video.mp4")

mog2 = openCv.createBackgroundSubtractorMOG2()

# Infinity loop for check all video
quitVideo = False
while quitVideo is False:
    ret, frame = cap.read() # Pick the if there are video return and the frame
    gray = openCv.cvtColor(frame, openCv.COLOR_BGR2GRAY) # Set frame to gray
    mask = mog2.apply(gray) # Set a mask to frame     
    retval, thresh = openCv.threshold(mask, 200, 255, openCv.THRESH_BINARY) # Thresold to clean the shadows, set gray to black. Pick if there are return and the thresold
    
    # Clear the noise
    kernel = openCv.getStructuringElement(openCv.MORPH_ELLIPSE, (5, 5))
    opening = openCv.morphologyEx(thresh, openCv.MORPH_OPEN, kernel, iterations = 2)

    dilation = openCv.dilate(opening, kernel, iterations = 8) # Expand the objects in the frame
    closing = openCv.morphologyEx(dilation, openCv.MORPH_CLOSE, kernel, iterations = 8) # Fill the noise inside an object

    openCv.imshow("frame", frame) # Show the frame picked
    openCv.imshow("gray", gray) # Show the frame picked
    openCv.imshow("mask", mask) # Show the frame picked
    openCv.imshow("thresh", thresh) # Show the frame picked
    openCv.imshow("opening", opening) # Show the frame picked
    openCv.imshow("dilation", dilation) # Show the frame picked
    openCv.imshow("closing", closing) # Show the frame picked
    
    if openCv.waitKey(30) & 0xFF == ord('q'):
        quitVideo = True

cap.release()
openCv.destroyAllWindows()