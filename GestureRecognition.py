import numpy as np
import cv2
import math

def angle(s,e,f): # Funcion proporcionada por el profesor para el calculo del angulo que forman los defectos de convexidad
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
	
def on_change_angle(value): # Funcion invocada por el slider de angulo para controlar el angulo maximo que puede tener los defectos de convexidad
	global max_angle
	if (value > 0):
		max_angle = value

def on_change_backSub(value): # Funcion invocada por el slider del MOG2Threshold que reinicia el modelo con un varThreshold diferente
	global backSub
	if (value > 0):
		backSub = cv2.createBackgroundSubtractorMOG2(history = 2, detectShadows = True,varThreshold=value)

def on_change_area(value): # Funcion invocada por el slider para cambiar el area minima que define el tener un dedo solo levantado
	global area_threshold
	if (value > 0):
		area_threshold = value

def on_change_process_rectangle_y(value): # Funcion invocada por el slider para cambiar la coordenada y del punto superior izquierdo del cuadrado de region de interes
	global pt1
	if (value > 0 and (value <= pt2[1])):
		pt1 = (pt1[0],value)
		cv2.setTrackbarMax('ProcessRectangle_Height', 'tools' ,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-pt1[1]))
		cv2.setTrackbarPos('ProcessRectangle_Height', 'tools' ,pt2[1]-pt1[1])
def on_change_process_rectangle_x(value): # Funcion invocada por el slider para cambiar la coordenada x del punto superior izquierdo del cuadrado de region de interes
	global pt1
	if (value > 0 and value <= pt2[0]):
		pt1 = (value,pt1[1])
		cv2.setTrackbarMax('ProcessRectangle_Width', 'tools' ,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)-pt1[0]))
		cv2.setTrackbarPos('ProcessRectangle_Width', 'tools' ,pt2[0]-pt1[0])
def on_change_process_rectangle_height(value): # Funcion invocada por el slider para cambiar la altura del cuadrado de region de interes
	global pt2
	if (value > 0):
		pt2 = (pt2[0],value+pt1[1])
def on_change_process_rectangle_width(value): # Funcion invocada por el slider para cambiar la anchura del cuadrado de region de interes
	global pt2
	if (value > 0):
		pt2 = (value+pt1[0],pt2[1])

global pt1	# Variables globales para modificar con los sliders el tamaño del cuadrado de interes
global pt2	
opened = False
i = 0
while (not opened and i <= 5): # Bucle para recorrer todos los /dev/video por si la camara se encuentra en otro device que no sea el primero
	cap = cv2.VideoCapture(i)
	if cap.read()[0]:
		opened = True
	i +=1
	if (i == 6):
		print ("No hay ninguna camara disponible")
		exit(0)

backSub = cv2.createBackgroundSubtractorMOG2(history = 2, detectShadows = True,varThreshold=50) # Se crea el modelo substractor de fondo, con historial = 2, deteccion de sombras y con threshold para filtrar ruido en nuestro caso
max_angle = 90
area_threshold = 60

pt1 = (850,250) # Tamaño tipico para camara de 1280x720, en caso de que sea de menor tamaño se colocara al minimo para evitar asserts
pt2 = (1150,350)
if (pt2[0] > cap.get(cv2.CAP_PROP_FRAME_WIDTH) or pt2[1] > cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
	pt1 = (0,0)
	pt2 = (1,1)


#Todas las ventanas y los sliders que se van a necesitar para ajustar el procesamiento
cv2.namedWindow('roi')
cv2.namedWindow('frame')
cv2.namedWindow('tools',cv2.WINDOW_NORMAL)
cv2.moveWindow('roi',120,300)
cv2.moveWindow('frame',600,100)
cv2.createTrackbar('Angle', 'tools' , 90, 180, on_change_angle)
cv2.createTrackbar('MOG2Threshold', 'tools' , 67, 100, on_change_backSub)
cv2.createTrackbar('AreaThreshold', 'tools' , 60, 100, on_change_area)
cv2.createTrackbar('ProcessRectangle_X', 'tools' , 850, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), on_change_process_rectangle_x)
cv2.createTrackbar('ProcessRectangle_Width', 'tools' , 300, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)-pt1[0]), on_change_process_rectangle_width)
cv2.createTrackbar('ProcessRectangle_Y', 'tools' , 250, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), on_change_process_rectangle_y)
cv2.createTrackbar('ProcessRectangle_Height', 'tools' , 250, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-pt1[1]), on_change_process_rectangle_height)
cv2.setWindowTitle('roi', 'roi - [p] pausar learning rate [q] abortar')

# Comenzamos con el learning rate a automatico para filtrar el fondo
learning_rate = -1

#Comenzamos con la lectura infinita de la feed de la camara
while (True):
	ret,frame=cap.read()

	if not ret:
		exit(0)
	
	#Es necesario invertir tanto horizontalmente como verticalmente el frame
	frame = cv2.flip(frame,1)
	
	roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()	#El roi será el cuadrado de interes
	cv2.rectangle(frame,pt1,pt2,(255,0,0))	#Dibujamos el cuadrado de interes en el frame principal
	fgMask = backSub.apply(roi,None,learning_rate) #Aplicamos el modelo MOG2 a la region de interes
	fgMask_3_channel = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR) #Es necesario pasar la mascara a canal BGR para poder unirlo con otra imagen con las ventanas de opencv
	contours, hierarchy = cv2.findContours(fgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:] #Buscamos los contornos de la mascara de 0 y 1, cogiendo solo los contornos externos y guardando solo los puntos finales de las lineas
	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) #Ordenamos los contornos de mayor area a menor ya que solo queremos la mano y no el ruido
	if (len(contours) > 0 and cv2.contourArea(contours[0]) > 100.0): # Evitamos realizar el computo en frames que no tengan ningun contorno o que sean contornos inutiles
		cv2.drawContours(roi, contours[0], -1, (0,255,0),3) # Dibujamos el contorno de mayor area
		cnt = contours[0]
		hull = cv2.convexHull(cnt,returnPoints = False) # Calculamos el covexhull del contorno, tomando solo los indices
		hull[::-1].sort(axis=0) # Lo ordenamos porque dependiendo de la version de opencv puede dar problemas
		defects = cv2.convexityDefects(cnt,hull) # Calculamos los defectos de convexidad con el contorno y la mascara 
		rect = cv2.boundingRect(cnt) # Calculamos el bounding rect del contorno de mayor area
		pt_1 = (rect[0],rect[1])
		pt_2 = (rect[0]+rect[2],rect[1]+rect[3])
		cv2.rectangle(roi,pt_1,pt_2,(0,0,255),3) # Lo dibujamos
		valid_defect_count = 0 # Contador de defectos de convexidad importantes
		if (defects is not None):
			for k in range(len(defects)):
				#print(len(defects))
				s,e,f,d = defects[k,0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])
				depth = d/256.0
				#print(depth)
				ang = angle(start,end,far) #Funcion proporcionada para obtener el angulo que forman 3 defectos de convexidad
				if (ang < max_angle and depth > 20): # Filtramos los que superen el angulo del slider y tengan una profundidad minima (ruido)
					cv2.line(roi,start,end,[255,0,0],2)
					cv2.circle(roi,far,5,[0,0,255],-1)
					valid_defect_count += 1
			cv2.putText(roi,str(valid_defect_count),(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,2) # Una vez recorrido todos los defectos, mostramos la cuenta de los que forman parte de la mano
			if (valid_defect_count == 0):
				result = (((rect[2]*rect[3]) % cv2.contourArea(contours[0]))/cv2.contourArea(contours[0])*100) # Calculamos el porcentaje de area del bounding rect frente al del contorno de la mano para detectar el caso de un solo dedo o ningun dedo
				#print(((rect[2]*rect[3]) % cv2.contourArea(contours[0]))/cv2.contourArea(contours[0])*100)
				if (result > area_threshold):
					cv2.putText(roi,'Un Dedo',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
				else:
					cv2.putText(roi,'Ningun Dedo',(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
			#Por cada cantidad de defectos validos mostramos un mensaje diferente, 4 = 5 dedos, 3 = 4 dedos etc...
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
	roi_mask = np.vstack((roi,fgMask_3_channel))	#Unimos verticalmente el roi y la fgmask para mejor aspecto visual
	cv2.imshow('frame',frame) #Mostramos el frame principal del momento
	cv2.imshow('roi',roi_mask) #Mostramos la union del roi y el mask del momento

	#Esperamos por si se recibe una entrada de teclado que queremos tratar
	keyboard = cv2.waitKey(40)
	if keyboard & 0xFF == ord('q'):
		break
	if keyboard & 0xFF == ord('p'): #para el learning rate y actualiza el texto de la ventana
		if (learning_rate != 0):
			cv2.setWindowTitle('roi', 'roi - Learning Rate pausado [r] para reanudarlo')
		learning_rate = 0
	if keyboard & 0xFF == ord('r'):
		if (learning_rate != -1): #reanuda el learning rate y actualiza el texto de la ventana
			cv2.setWindowTitle('roi', 'roi - [p] pausar learning rate [q] abortar')
		learning_rate = -1

#Liberamos la camara y destruimos todas las ventanas
cap.release()
cv2.destroyAllWindows()

