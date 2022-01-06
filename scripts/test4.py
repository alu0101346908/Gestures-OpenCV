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
	
def on_change(value):
	global max_angle
	if (value > 0):
		max_angle = value

def on_change_backSub(value):
	global backSub
	if (value > 0):
		backSub = cv2.createBackgroundSubtractorMOG2(history = 2, detectShadows = True,varThreshold=value)

def on_change_area(value):
	global area_threshold
	if (value > 0):
		area_threshold = value

	
cap = cv2.VideoCapture(2)
backSub = cv2.createBackgroundSubtractorMOG2(history = 2, detectShadows = True,varThreshold=50)
max_angle = 90
area_threshold = 60
if not cap.isOpened:
    print ("Unable to open file")
    exit(0)
	#pendiente hacerlo relativo al tamaño de la feed de la camara
pt1 = (850,250)
pt2 = (1150,500)
#pt1 = (1300,250)
#pt2 = (1700,700)
cv2.namedWindow('roi')
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.moveWindow('roi',120,300)
cv2.moveWindow('frame',600,100)
cv2.createTrackbar('Angle', 'roi' , 90, 180, on_change)
cv2.createTrackbar('MOG2Threshold', 'roi' , 67, 100, on_change_backSub)
cv2.createTrackbar('AreaThreshold', 'roi' , 60, 100, on_change_area)
learning_rate = -1
while (True):
	ret,frame=cap.read()
	if not ret:
		exit(0)
	
	frame = cv2.flip(frame,1)
	
	roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()
	cv2.rectangle(frame,pt1,pt2,(255,0,0))
	imS = cv2.resize(frame, (1280, 720))

	gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
	fgMask = backSub.apply(roi,None,learning_rate)
	fgMask_3_channel = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
	contours, hierarchy = cv2.findContours(fgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
	if (len(contours) > 0 and cv2.contourArea(contours[0]) > 100.0):
		cv2.drawContours(roi, contours[0], -1, (0,255,0),3)
		for i in range (1):
			cnt = contours[0]
			hull = cv2.convexHull(cnt,returnPoints = False)
			hull[::-1].sort(axis=0)
			defects = cv2.convexityDefects(cnt,hull)
			rect = cv2.boundingRect(cnt)
			pt_1 = (rect[0],rect[1])
			pt_2 = (rect[0]+rect[2],rect[1]+rect[3])
			cv2.rectangle(roi,pt_1,pt_2,(0,0,255),3)
			valid_defect_count = 0
			if (defects is not None):
				for k in range(len(defects)):
					#print(len(defects))
					s,e,f,d = defects[k,0]
					start = tuple(cnt[s][0])
					end = tuple(cnt[e][0])
					far = tuple(cnt[f][0])
					depth = d/256.0
					#print(depth)
					ang = angle(start,end,far)
					if (ang < max_angle and depth > 10):
						cv2.line(roi,start,end,[255,0,0],2)
						cv2.circle(roi,far,5,[0,0,255],-1)
						valid_defect_count += 1
				cv2.putText(roi,str(valid_defect_count),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
				if (valid_defect_count == 0):
					result = (((rect[2]*rect[3]) % cv2.contourArea(contours[0]))/cv2.contourArea(contours[0])*100)
					print(((rect[2]*rect[3]) % cv2.contourArea(contours[0]))/cv2.contourArea(contours[0])*100)
					if (result > area_threshold):
						cv2.putText(roi,'Un Dedo',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
					else:
						cv2.putText(roi,'Ningun Dedo',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
				if (valid_defect_count == 1):
					cv2.putText(roi,'Dos Dedos',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
				if (valid_defect_count == 2):
					cv2.putText(roi,'Tres Dedos',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
				if (valid_defect_count == 3):
					cv2.putText(roi,'Cuatro Dedos',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
				if (valid_defect_count == 4):
					cv2.putText(roi,'Cinco Dedos',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
				#print("contour: " + str(cv2.contourArea(contours[0])))
				#print("rect: " + str(rect[2]*rect[3]))
	roi_mask = np.vstack((roi,fgMask_3_channel))
	cv2.imshow('frame',frame)
	cv2.imshow('roi',roi_mask)
	keyboard = cv2.waitKey(40)
	if keyboard & 0xFF == ord('q'):
		break
	if keyboard & 0xFF == ord('p'):
		learning_rate = 0
	if keyboard & 0xFF == ord('r'):
		learning_rate = -1

cap.release()
cv2.destroyAllWindows()

