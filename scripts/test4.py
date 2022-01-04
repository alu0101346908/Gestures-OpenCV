import numpy as np
import cv2


cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open file")
    exit(0)
pt1 = (850,250)
pt2 = (1150,500)
learning_rate = -1
while (True):
	ret,frame=cap.read()
	if not ret:
		exit(0)
	
	frame = cv2.flip(frame,-1)
	
	roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()
	fgMask = backSub.apply(roi,None,learning_rate)
	cv2.rectangle(frame,pt1,pt2,(255,0,0))
	cv2.imshow('frame',frame)
	cv2.imshow('ROI',roi)
	cv2.imshow('fgMask',fgMask)
	keyboard = cv2.waitKey(40)
	if keyboard & 0xFF == ord('q'):
		break
	if keyboard & 0xFF == ord('p'):
		learning_rate = 0.0005
	if keyboard & 0xFF == ord('r'):
		learning_rate = -1

cap.release()
cv2.destroyAllWindows()

