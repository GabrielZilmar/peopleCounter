import numpy as np
import cv2 as openCv

# Function to return the rectangle's center
def getCenter(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Function to count people
def countPeople():
    cap = openCv.VideoCapture("video.mp4")

    mog2 = openCv.createBackgroundSubtractorMOG2()

    posLine = 140
    offset = 35
    xy1 = (70, posLine)
    xy2 = (300, posLine)
    detects = [] # Matrix with the position of the peoples

    total = 0
    goingUp = 0
    goingDown = 0

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
            (x, y, w, h) = openCv.boundingRect(cnt) # Get contours area
            area = openCv.contourArea(cnt) # Set the people's area

            # Check if the area is considerably
            if int(area) > 3000:
                center = getCenter(x, y, w, h) # Set the people's center

                openCv.putText(frame, str(count), (x+5, y+15), openCv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) # Count how many people are in the same frame

                openCv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Set the people's contour in the original video
                openCv.circle(frame, center, 4, (0, 0, 255), -1) # Set the people's center in the original video
                
                # Add the positions
                if len(detects) <= count:
                    detects.append([])
                if center[1] > posLine-offset and center[1] < posLine+offset:
                    detects[count].append(center)

                count += 1

        # If there aren't people in the frame, clear the detects
        if len(contours) == 0:
            detects.clear()
        if len(detects) > 0:
            for detect in detects:
                for (c, l) in enumerate(detect):
                    # Verify if the person is going to top
                    if detect[c-1][1] < posLine and l[1] > posLine:
                        detect.clear()
                        goingUp += 1
                        total += 1
                        openCv.line(frame, xy1, xy2, (0, 255, 0), 5)
                        continue
                    # Verify if the person is going to down
                    if detect[c-1][1] > posLine and l[1] < posLine:
                        detect.clear()
                        goingDown += 1
                        total += 1
                        openCv.line(frame, xy1, xy2, (0, 255, 0), 5)
                        continue
                    
                    # Tracking line
                    if c > 0:
                        openCv.line(frame, detect[c-1], l, (0, 0, 255), 1)

        openCv.putText(frame, "Total: " + str(total), (10, 20), openCv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        openCv.putText(frame, "Going Up: " + str(goingUp), (10, 40), openCv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        openCv.putText(frame, "Going Down: " + str(goingDown), (10, 60), openCv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        openCv.putText(frame, "PRESS Q TO EXIT", (240, 270), openCv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        openCv.imshow("frame", frame) # Show the frame picked
        # openCv.imshow("closing", closing) # Show the frame picked processed
        
        if openCv.waitKey(30) & 0xFF == ord('q'):
            quitVideo = True

    cap.release()
    openCv.destroyAllWindows()

countPeople()