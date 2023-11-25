import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
already_trained = 0
display_all = 1

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

def load_dataset(image_folder, label_file_path):
    # Read the labels from the text file
    labels_df = pd.read_csv(label_file_path, delimiter=' ', header=None, names=['File', 'Label', 'Timestamp'])

    data = []
    labels = []
    imagenum = []

    # Loop through each row in the labels dataframe
    for index, row in labels_df.iterrows():
        # Construct the full path to the image
        image_path = os.path.join(image_folder, row['File'])

        # Read the image using PIL
        image = Image.open(image_path)

        # Extract color histogram features from the image
        hist = extract_color_histogram(image)

        # Append the features to the 'data' list
        data.append(hist)

        # Append the label to the 'labels' list
        labels.append(row['Label'])
        imagenum.append(row['File'])
        
    # Return the collected features and labels as NumPy arrays
    return np.array(data), np.array(labels), np.array(imagenum)

# Replace 'frames' and 'labels.txt' with your actual folder names and label file
image_folder = 'first_test'
label_file_path = 'label_text'


if already_trained == 0:
	# Load the dataset
	X, y, fns = load_dataset(image_folder, label_file_path)
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# Train the classifier
	classifier = SVC(kernel='linear', C=1.0)
	classifier.fit(X_train, y_train)
	# Evaluate the classifier
	y_pred = classifier.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {accuracy}")

	print("Classification Report:")
	print(classification_report(y_test, y_pred))

	# Save the trained classifier to a file
	joblib.dump(classifier, 'jokers_classifier_model.pkl')

def sort_dataframe_by_filename(df):
    # Extract numeric part from filenames and create a new column
    df['numeric_part'] = df.iloc[:, 0].str.extract('(\d+)').astype(int)

    # Sort DataFrame based on the new column
    sorted_df = df.sort_values(by='numeric_part')

    # Drop the temporary column
    sorted_df = sorted_df.drop(columns=['numeric_part'])

    return sorted_df
    


if display_all:

	classifier = joblib.load('jokers_classifier_model.pkl')
	# Load the dataset again
	X, y_true, fns = load_dataset(image_folder, label_file_path)
	
	print(y_true[0:10])
	print(y_true[10:30])
	# Make predictions
	y_pred = classifier.predict(X)

	# Create a DataFrame with original labels, predicted labels, and correctness
	results_df = pd.DataFrame({
	    'File': fns,
	    'True_Label': y_true,
	    'Predicted_Label': y_pred,
	    'Correct': (y_true == y_pred)
	    })
        
        
	sorted_df = sort_dataframe_by_filename(results_df)
	#results_df['numeric_part'] = sorted_df.iloc[:, 0].str.extract('(\d+)').astype(int)
        # Sort DataFrame based on the new column
	#sorted_df = sorted_df.sort_values(by='numeric_part')

        # Drop the temporary column
	#sorted_df = sorted_df.drop(columns=['numeric_part'])
	# Merge with the original DataFrame to include timestamps
	#results_df = pd.merge(results_df, original_df[['File', 'Timestamp']], on='File', how='left')

	# Sort the DataFrame by timestamps
	#results_df.sort_values(by='Timestamp', inplace=True)

	# Save the results to a text file
	sorted_df.to_csv('prediction_results_sorted', sep=' ', index=False)


