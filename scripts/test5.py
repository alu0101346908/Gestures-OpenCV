import numpy as np
import cv2


cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open cam")
    exit(0)

while (True):
    ret,frame=cap.read()
    if not ret:
	    exit(0)
    fgMask = backSub.apply(frame)
    #gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    ret,bw = cv2.threshold(fgMask,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(fgMask, contours, -1, (0,255,0),3)
    rect = cv2.boundingRect(contours[0])
    pt1 = (rect[0],rect[1])
    pt2 = (rect[0]+rect[2],rect[1]+rect[3])

    cv2.rectangle(fgMask,pt1,pt2,(0,0,255),3)

    cv2.imshow('frame',frame)
    cv2.imshow('Foreground Mask',fgMask)

    keyboard = cv2.waitKey(1)
    if keyboard & 0xFF == ord('q'):
	    break

cap.release()
cv2.destroyAllWindows()

