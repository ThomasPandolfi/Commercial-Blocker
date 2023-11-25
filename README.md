# Commercial-Blocker
TV Commercial Mute Automation System
Overview
This project is designed to automatically detect commercials on a TV screen, mute the audio, and unmute when the commercial ends. The system involves a hardware setup with an Arduino to control the TV's mute function and software scripts for capturing frames, labeling data, training an SVM classifier, implementing a memory-based method for continuous state assessment, and a live predictor for real-time applications.

Files
1. capture_frames.py
Purpose: Captures frames from the television and saves them in an ascending naming convention.
Methodology: Suitable for testing and data collection for model training.
2. create_labels.py
Purpose: Creates a text file with three columns: file name, label, and timestamp.
Methodology: Organizes data into folders based on labeling and generates the necessary input for model training.
3. create_classifier.py
Purpose: Trains an SVM classifier on frames reduced to HSV space, tests it on the dataset, and sorts data by framenumber.
Note: Required to prepare data for the next script.
4. memory_method.py
Purpose: Implements a model for state assessment based on SVM predictions, time decay functions, and transition logic.
Methodology: Considers the SVM output values over a specified time period to predict state transitions.
Note: The end of the code contains a parameter analysis section to fine-tune the model.
5. live_predictor.py
Purpose: Provides a live prediction for real-time applications using preloaded SVM classifier and methodology from memory_method.py.
Note: Includes space for interfacing with the Arduino serial monitor to send an infrared signal when state transitions occur.
Getting Started
Capture Frames:

Run capture_frames.py to collect frames from the TV.
Label Data:

Organize frames into labeled folders.
Run create_labels.py to generate a text file with labeling and timestamps.
Train SVM Classifier:

Run create_classifier.py to train and test the SVM classifier.
Implement Memory Method:

Run memory_method.py to assess states and control TV mute based on SVM predictions and time decay functions.
Live Prediction:

Run live_predictor.py for real-time applications.
Interface with the Arduino serial monitor for TV control when state transitions occur.
Notes
Adjust parameters in memory_method.py and live_predictor.py for fine-tuning the model.
Ongoing work to optimize the system for accurate state detection.
