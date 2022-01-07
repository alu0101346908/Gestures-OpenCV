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

def on_change_process_rectangle_y(value):
	global pt1
	if (value > 0 and (value <= pt2[1])):
		pt1 = (pt1[0],value)
		cv2.setTrackbarMax('ProcessRectangle_Height', 'tools' ,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-pt1[1]))
		cv2.setTrackbarPos('ProcessRectangle_Height', 'tools' ,pt2[1]-pt1[1])
def on_change_process_rectangle_x(value):
	global pt1
	if (value > 0 and value <= pt2[0]):
		pt1 = (value,pt1[1])
		cv2.setTrackbarMax('ProcessRectangle_Width', 'tools' ,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)-pt1[0]))
		cv2.setTrackbarPos('ProcessRectangle_Width', 'tools' ,pt2[0]-pt1[0])
def on_change_process_rectangle_height(value):
	global pt2
	if (value > 0):
		pt2 = (pt2[0],value+pt1[1])
def on_change_process_rectangle_width(value):
	global pt2
	if (value > 0):
		pt2 = (value+pt1[0],pt2[1])

global pt1
global pt2
opened = False
i = 0
while (not opened and i <= 5):
	cap = cv2.VideoCapture(i)
	if cap.read()[0]:
		opened = True
	i +=1
	if (i == 6):
		print ("No hay ninguna camara disponible")
		exit(0)
backSub = cv2.createBackgroundSubtractorMOG2(history = 2, detectShadows = True,varThreshold=50)
max_angle = 90
area_threshold = 60

pt1 = (850,250)
pt2 = (1150,350)
if (pt2[0] > cap.get(cv2.CAP_PROP_FRAME_WIDTH) or pt2[1] > cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
	pt1 = (0,0)
	pt2 = (1,1)

#pt1 = (1300,250)
#pt2 = (1700,700)
cv2.namedWindow('roi')
cv2.namedWindow('frame')
cv2.namedWindow('tools',cv2.WINDOW_NORMAL)
cv2.moveWindow('roi',120,300)
cv2.moveWindow('frame',600,100)
cv2.createTrackbar('Angle', 'tools' , 90, 180, on_change)
cv2.createTrackbar('MOG2Threshold', 'tools' , 67, 100, on_change_backSub)
cv2.createTrackbar('AreaThreshold', 'tools' , 60, 100, on_change_area)
cv2.createTrackbar('ProcessRectangle_X', 'tools' , 850, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), on_change_process_rectangle_x)
cv2.createTrackbar('ProcessRectangle_Width', 'tools' , 300, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)-pt1[0]), on_change_process_rectangle_width)
cv2.createTrackbar('ProcessRectangle_Y', 'tools' , 250, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), on_change_process_rectangle_y)
cv2.createTrackbar('ProcessRectangle_Height', 'tools' , 250, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-pt1[1]), on_change_process_rectangle_height)
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
					if (ang < max_angle and depth > 20):
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
		if (learning_rate != 0):
			cv2.setWindowTitle('roi', 'roi - Learning Rate pausado')
		learning_rate = 0
	if keyboard & 0xFF == ord('r'):
		if (learning_rate != -1):
			cv2.setWindowTitle('roi', 'roi')
		learning_rate = -1

cap.release()
cv2.destroyAllWindows()

