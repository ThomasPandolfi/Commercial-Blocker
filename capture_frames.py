import cv2
import matplotlib.image
cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture(0)
import numpy as np

import time

framename = 3000
while 1:
	time.sleep(1)
	frame = np.flip(cap.read()[1],2)
	matplotlib.image.imsave('first_test/' + str(framename) + '.jpg', frame)
	framename += 1

