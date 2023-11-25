
import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

import serial
import time
#arduino = serial.Serial(port = 'COM3', timeout=0)


import cv2
import matplotlib.image
cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture(0)
import numpy as np

import time


def extract_color_histogram(image):
	# Convert the image to the HSV color space
	hsv = image.convert('HSV')
	# Split the image into separate color channels
	h, s, v = hsv.split()
	# Calculate the histogram for each color channel
	hist_h = np.array(h.histogram(), dtype=np.float64)
	hist_s = np.array(s.histogram(), dtype=np.float64)
	hist_v = np.array(v.histogram(), dtype=np.float64)
	# Normalize the histograms
	hist_h /= hist_h.sum() + 1e-6
	hist_s /= hist_s.sum() + 1e-6
	hist_v /= hist_v.sum() + 1e-6
	# Concatenate the histograms into a single feature vector
	hist = np.concatenate([hist_h, hist_s, hist_v])
	return hist




def weighted_mean(ar):
	weights = np.arange(1, len(ar) + 1)
	#print(weights)
	weighted_m = np.average(ar,weights=weights, axis= 0)
	return weighted_m
 




def calculate_switch_probability(predictions, recent_time_decay, delay, t, previous_state):
	recent_predictions = predictions[-delay:]  # Consider the last 'delay' predictions
	memory_prediction = np.mean(recent_predictions)
	memory_prediction = weighted_mean(recent_predictions)
	recent_time_decay = np.mean(recent_time_decay) #this does nothing
	if previous_state == 1:
		state_probability = previous_state - memory_prediction
	else:
		state_probability = memory_prediction - previous_state
	return state_probability * (1 - recent_time_decay), state_probability, 1-recent_time_decay


# Create a custom time decay function
def exponential_decay(t, P0_state1, P0_state2, state, K_state1 = -np.log(0.05) / 150, K_state2 = -np.log(0.005) / 150):
	if state == 1:
		return P0_state1 * np.exp(-K_state1 * t)
	else:
		return P0_state2 * np.exp(-K_state2 * t)


#Necessary Variables
framename = 50 #Assume that we start midframe of a show. this issnt important for prediction, but for logging it might be useful
t = 0 #current time value since last change, resets every switch from commercial to show
previous_state = 1 #Lets assume we start during a show
classifier = joblib.load('image_classifier_model.pkl') #Grab the classifier
svm_labels = [] #will append a 1 or 0 for each frame, whatever the SVM outputs
predicted_labels = [] #what the model thinks the state is

initial_state = 0
cutoff = 0.9
delay = 10



from PIL import Image
import matplotlib
#There might be black frames, assume the first frame is black
frame = matplotlib.image.imread('black.png')

while 1: #loop foreva!
	
	#keep grabbing frames until one is viable
	#while np.mean((frame[:,:,0] +frame[:,:,1] + frame[:,:,2])/3) < 0.01:
		
	while 1:
		time.sleep(1)
		frame = np.flip(cap.read()[1],2)
		if np.mean((frame[:,:,0] +frame[:,:,1] + frame[:,:,2])/3) == 0.00:
			print('black')
		else:
			break
		
	#Image is good, save and increment frame number	
	matplotlib.image.imsave('first_test/' + str(framename) + '.jpg', frame)
	
	#Frame is the raw rgb of the image
	# Read the image using PIL
	image = Image.open('first_test/' + str(framename) + '.jpg')
	framename += 1
        # Extract color histogram features from the image
	hist = extract_color_histogram(image)

	#print('we got here')
	#Predict what class its in, append to predicted labels array
	y_pred = classifier.predict(hist.reshape(1,-1))
	
	print(f"SVM value: {y_pred}")
	svm_labels.append(int(y_pred))
	
	
	#we need to establish a certain number of frames to grab before actually doing stuff, lets do the delay period as the necessary length
	if len(predicted_labels) >= delay: #if the length of predicted labels is acceptable. methinks I could use svm_labels here as well
		A, B, C = calculate_switch_probability(svm_labels, exponential_decay(t, 1, 1, previous_state), delay, 0, previous_state) #calculate probability
		if A > cutoff:
			previous_state = int(not(previous_state))
			t = 0
			#insert sending mute button code
			print('SWITCHING')
						
		predicted_labels.append(previous_state) #might as well do it after the logic	
			

			
		t = t + 1
		print(f"framenumber: {framename}, State: {previous_state}, AValue: {A}")
		
	else:
		print(f"framenumber: {framename}, not yet long enough")
		predicted_labels.append(y_pred)
		
	




