import numpy as np
import cv2 as openCv

# Function to return the rectangle's center
def getCenter(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = openCv.VideoCapture("video.mp4")

mog2 = openCv.createBackgroundSubtractorMOG2()

posLine = 150
offset = 30

xy1 = (20, posLine)
xy2 = (300, posLine)

# Infinity loop for check all video
quitVideo = False
while quitVideo is False:
    ret, frame = cap.read() # Pick if there are video return and the frame returned
    gray = openCv.cvtColor(frame, openCv.COLOR_BGR2GRAY) # Set frame to gray
    mask = mog2.apply(gray) # Set a mask to frame     
    retval, thresh = openCv.threshold(mask, 200, 255, openCv.THRESH_BINARY) # Thresold to clean the shadows, set gray to black. Pick if there are return and the thresold
    
    # Clear the noise
    kernel = openCv.getStructuringElement(openCv.MORPH_ELLIPSE, (5, 5))
    opening = openCv.morphologyEx(thresh, openCv.MORPH_OPEN, kernel, iterations = 2) # Round the objects

    dilation = openCv.dilate(opening, kernel, iterations = 8) # Expand the objects in the frame
    closing = openCv.morphologyEx(dilation, openCv.MORPH_CLOSE, kernel, iterations = 8) # Fill the noise inside an object
    contours, hierarchy = openCv.findContours(closing, openCv.RETR_TREE, openCv.CHAIN_APPROX_SIMPLE) # Contour the objects

    openCv.line(frame, xy1, xy2, (255, 0, 0), 3) # Set a line in the original video
    # Set offset's lines in the original video
    openCv.line(frame, (xy1[0], posLine-offset), (xy2[0], posLine-offset), (255, 0, 255), 2) 
    openCv.line(frame, (xy1[0], posLine+offset), (xy2[0], posLine+offset), (255, 0, 255), 2)

    # Walk in the contours
    count = 0
    for cnt in contours:
        # Get contours area
        (x, y, w, h) = openCv.boundingRect(cnt)
        area = openCv.contourArea(cnt) # Set the people's area

        # Check if the area is considerably
        if int(area) > 3000:
            center = getCenter(x, y, w, h) # Set the people's center

            openCv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Set the people's contour in the original video
            openCv.circle(frame, center, 4, (0, 0, 255), -1) # Set the people's center in the original video

    openCv.imshow("frame", frame) # Show the frame picked
    openCv.imshow("closing", closing) # Show the frame picked
    
    if openCv.waitKey(30) & 0xFF == ord('q'):
        quitVideo = True

cap.release()
openCv.destroyAllWindows()