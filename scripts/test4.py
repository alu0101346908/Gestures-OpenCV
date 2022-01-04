import numpy as np
import cv2


cap = cv2.VideoCapture(2)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open file")
    exit(0)
	#pendiente hacerlo relativo al tamaño de la feed de la camara
pt1 = (1300,250)
pt2 = (1700,700)
learning_rate = -1
while (True):
	ret,frame=cap.read()
	if not ret:
		exit(0)
	
	frame = cv2.flip(frame,1)
	
	roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()
	fgMask = backSub.apply(roi,None,learning_rate)
	cv2.rectangle(frame,pt1,pt2,(255,0,0))
	imS = cv2.resize(frame, (1280, 720))

	gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
	contours, hierarchy = cv2.findContours(fgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	cv2.drawContours(roi, contours, -1, (0,255,0),3)
	for i in range(len(contours)):
		hull = cv2.convexHull(contours[i])
		cv2.drawContours(roi, [hull], -1, (255,0,0),3)

	cv2.imshow('frame',imS)
	cv2.imshow('ROI',roi)
	cv2.imshow('fgMask',fgMask)
	keyboard = cv2.waitKey(40)
	if keyboard & 0xFF == ord('q'):
		break
	if keyboard & 0xFF == ord('p'):
		learning_rate = 0
	if keyboard & 0xFF == ord('r'):
		learning_rate = -1

cap.release()
cv2.destroyAllWindows()

