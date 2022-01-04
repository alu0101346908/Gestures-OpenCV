import numpy as np
import cv2
import math

def angle(s,e,f):
    v1 = [s[0]-f[0],s[1]-f[1]]
    v2 = [e[0]-f[0],e[1]-f[1]]
    ang1 = math.atan2(v1[1],v1[0])
    ang2 = math.atan2(v2[1],v2[0])
    ang = ang1 - ang2
    if (ang > np.pi):
        ang -= 2*np.pi
    if (ang < -np.pi):
        ang += 2*np.pi
    return ang*180/np.pi
	

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open file")
    exit(0)
	#pendiente hacerlo relativo al tamaño de la feed de la camara
pt1 = (850,250)
pt2 = (1150,500)
#pt1 = (1300,250)
#pt2 = (1700,700)
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
	if (len(contours) > 0):
		for i in range (len(contours)):
			cnt = contours[i]
			hull = cv2.convexHull(cnt,returnPoints = False)
			hull[::-1].sort(axis=0)
			defects = cv2.convexityDefects(cnt,hull)
			if (defects is not None):
				for k in range(len(defects)):
					s,e,f,d = defects[k][0]
					start = tuple(cnt[s][0])
					end = tuple(cnt[e][0])
					far = tuple(cnt[f][0])
					depth = d/256.0
					print(depth)
					ang = angle(start,end,far)
					cv2.line(roi,start,end,[255,0,0],2)
					cv2.circle(roi,far,5,[0,0,255],-1)


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

