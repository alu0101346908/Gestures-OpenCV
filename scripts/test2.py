import numpy as np
import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened:
	print ("Unable to open cam")
	exit(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
	
out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
	

while (True):
	ret,frame=cap.read()
	cv2.imshow('frame',frame)

	out.write(frame)

	keyboard = cv2.waitKey(1)
	if keyboard & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()

