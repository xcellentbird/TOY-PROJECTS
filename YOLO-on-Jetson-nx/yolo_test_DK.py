import darknet
import numpy as np
import cv2
import time

# load configure file, dataset, pre-weights
net, cn, color = darknet.load_network('./cfg/yolov4.cfg', './cfg/coco.data', './yolov4.weights')
width = darknet.network_width(net)
height = darknet.network_height(net)
darknet_image = darknet.make_image(width, height, 3)

cap = cv2.VideoCapture(0) # video capture

print('capped!')        
prevTime = 0
fpss = []
while(cap.isOpened()):
        ret, frame = cap.read() # read capture video frame
	if ret: # ture: read frame successed, false: read frame failed
		
		#### get FPS ####
		curTime = time.time()
		sec = curTime - prevTime
		prevTime = curTime
		fps = 1/(sec)
		fpss.append(fps)
		avr_fps = sum(fpss)/len(fpss)
		str = "FPS : %0.1f" % avr_fps

		#### Image Preprocessing, Detect, Draw Box ####
		frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) # resize image
		frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) # change bgr image 2 rgb image
		darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
		r = darknet.detect_image(net, cn, darknet_image) # detect obj from image 
		image = darknet.draw_boxes(r, frame_resized, color) # draw box on the image
		image = cv2.resize(image, (len(frame[0])/4, len(frame)/4)) 

		cv2.putText(image, str, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255)) # put fps text on image
		cv2.imshow('show',image)
		for found in r: # for all detected obj
			label_name, prob, loc = found[0], found[1], found[2] # get label name, probablility, location
			print(label_name)
		if cv2.waitKey(1) != -1:
			break
	else:
		print('No Frame')
		break
cap.release()
cv2.destroyAllWindows()
