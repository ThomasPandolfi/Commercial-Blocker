
#Set start frame number to label
start_frame = 0
#end_frame = 
import os
import shutil
from matplotlib import pyplot as plt

# Set the path to your images folder
images_folder = 'frames'
labels_folder = 'labels'

# Create subdirectories for each label
label_0_folder = os.path.join(labels_folder, '0')
label_1_folder = os.path.join(labels_folder, '1')
label_2_folder = os.path.join(labels_folder, '2')



folders_to_index = [label_0_folder, label_1_folder]

#renamed to be 1 and 0, makes it much much easier to parse

with open('label_text', 'w') as output_file:
	for folder in folders_to_index:
		label = folder.split('/')[1]
		for image in os.listdir(folder):
			timestamp = image[0:-4]
			
			output_file.write(f"{image} {label} {timestamp}\n")
			

	
