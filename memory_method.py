import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import  numpy as np
import matplotlib.pyplot as plt
import serial
import time

#arduino = serial.Serial(port = 'COM3', timeout=0)
# Load your DataFrame (replace 'your_data.csv' with your actual file)
df = pd.read_csv('prediction_results_sorted_resetstamp', delimiter = ' ')
# Assuming you have a DataFrame df with 'Predicted_Label' and 'Reset_Timestamps' columns
predictions = df['Predicted_Label'].values
timestamps = df['Reset_Timestamps'].values

# Create a custom time decay function
def exponential_decay(t, P0_state1, P0_state2, state, K_state1 = -np.log(0.005) / 150, K_state2 = -np.log(0.005) / 150):
    if state == 1:
        return P0_state1 * np.exp(-K_state1 * t)
    else:
    	return P0_state2 * np.exp(-K_state2 * t)
    	

# Create a new feature using the time decay function
time_decay_values = np.array([exponential_decay(t,1,1,0) for t in timestamps])
# Combie classifier predictions and time decay values
combined_features = np.column_stack((predictions, time_decay_values))
# Train on the entire dataset
X_train, y_train = combined_features, df['True_Label']
# Simple rule-based system for probability estimation


def weighted_mean(ar):
    weights = np.arange(1, len(ar) + 1)
    #print(weights)
    weighted_m = np.average(ar,weights=weights, axis= 0)
    return weighted_m
 




def calculate_switch_probability(predictions, recent_time_decay, delay, t, previous_state):
    recent_predictions = predictions[t-delay:t]  # Consider the last 5 predictions
    #recent_time_decay = time_decay_values[t-delay:t]  # Corresponding time decay values
    # Calculate the probability based on the recent history
    #print(recent_predictions)
    
    memory_prediction = np.mean(recent_predictions)
    #memory_prediction = weighted_mean(recent_predictions)
    
    
    
    recent_time_decay = np.mean(recent_time_decay) #this does nothing
    if previous_state == 1:
    	state_probability = previous_state - memory_prediction
    else:
    	state_probability = memory_prediction - previous_state
    return state_probability * (1 - recent_time_decay), state_probability, 1-recent_time_decay



def test_run(y_train, combined_features, previous_state, cutoff, delay, P01, P02, K_state1, K_state2):
#previous_state = 0
#cutoff = 0.5
#delay = 4
    tester = []
#A B and C list are the fraction value of the calculator that is compared to cutoff
#if value is greater than some cutoff, it "switches" to the other state
#B and C are the state probabilities (think - if last 5 values were state 1, this would be 1, if last 5 values were 0, would be 0, if inbetween its an average)
#C is the time decay using predicted state amount
    A_list = []
    B_list = []
    C_list = []
    #P01 = 1
    #P02 = 1
    t = 0
    for i in range(delay, len(combined_features[:,0])):
        A, B, C = calculate_switch_probability(combined_features[:,0], exponential_decay(t, P01, P02, previous_state, K_state1, K_state2), delay, i, previous_state)
        #A, B, C = calculate_switch_probability(tester, exponential_decay(t, 1, 1, previous_state), delay, i, previous_state)
        
        if A > cutoff:
    	    previous_state = int(not(previous_state))
    	    t = 0
        tester.append(previous_state)
        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
        t = t + 1
    
    #Plotting stuff
    if 0:
	    plt.plot(tester, label ='predicted with memory')
	    plt.plot(y_train.iloc[delay:].values *0.9, label='true value')
	    plt.plot(combined_features[:,0] * 0.8, label = 'SVM output', linewidth = 0.2)
	    plt.legend()
	    plt.show()
	    plt.plot(A_list, label = 'A', linewidth = 0.4)
	    #plt.plot(B_list, label = 'B', linewidth = 0.4)
	    plt.plot(C_list, label = 'C', linewidth = 0.4)
	    plt.plot(y_train.iloc[delay:].values *0.9, label='true value')
	    plt.legend()
	    plt.show()
	    
    return(A,B,C, tester)
    
    
initial_state = 0
cutoff = 0.9
delay = 12


A,B,C, predict = test_run(y_train, combined_features, initial_state, cutoff, delay, 1,1,-np.log(0.005)/150, -np.log(0.005)/150)
from sklearn.metrics import accuracy_score, confusion_matrix


correctness_array = (predict == y_train.iloc[delay:].values)
#print(correctness_array)

total_accuracy = np.mean(correctness_array)
conf_matrix = confusion_matrix(y_train.iloc[delay:].values, predict)
true_positives = conf_matrix[1, 1]
false_negatives = conf_matrix[1, 0]
true_negatives = conf_matrix[0, 0]
false_positives = conf_matrix[0, 1]
# Calculate sensitivity (recall)
sensitivity = true_positives / (true_positives + false_negatives)

# Calculate selectivity (true negative rate)
selectivity = true_negatives / (true_negatives + false_positives)


print(f"Total Accuracy: {total_accuracy}")
print(f"Sensitivity (TP/TP+FN): {sensitivity}")
print(f"Selectivity (TN/TN+FP): {selectivity}")
#overhang = np.sum(1 * (predict > y_train.iloc[delay:].values))
#underhang = np.sum(1 * (predict < y_train.iloc[delay:].values))
#print(f"OverHang amount: {overhang}")
#print(f"UnderHang amount: {underhang}")
print(len(predict), len(y_train.iloc[delay:].values))
#plt.plot(0.8* (predict > y_train.iloc[delay:].values), label='Overhang')
#plt.plot(0.8* (predict < y_train.iloc[delay:].values), label='Underhang')



#plt.plot(y_train.iloc[delay:].values, linewidth = 0.8, label='ActualValue')
#plt.plot(predict, linewidth = 0.8, label='Prediction')
#plt.legend()
#plt.show()

initial_state = 0
cutoff = 0.9
delay = 12
P01 = 1
P02 = 1
K_state1 = -np.log(0.005)/150
K_state2 = -np.log(0.005)/150



emp_cutoff = np.linspace(.5,1,22)
emp_delay = np.linspace(3,15,13).astype('int')
emp_P01 = np.linspace(0.5, 1, 5)
emp_P02 = np.linspace(0.5, 1, 5)
emp_K_state1 = np.linspace(0.005, 0.05, 15)
emp_K_state2 = np.linspace(0.005, 0.05, 15)

emp_TP =  []
emp_FN =  []
emp_TN =  []
emp_FP =  []
emp_SEN =  []
emp_SEL =  []


Lota = 0
print('Total iterations I need = ' + str(22*13*6*6*18*18))
for cutoff in emp_cutoff:
	print(Lota)
	for delay in emp_delay:
		for P01 in emp_P01:
			for P02 in emp_P02:
				for K_state1 in emp_K_state1:
					for K_state2 in emp_K_state2:
						
						A,B,C, predict = test_run(y_train, combined_features, initial_state, cutoff, delay, P01, P02, K_state1, K_state2)
						conf_matrix = confusion_matrix(y_train.iloc[delay:].values, predict)
						true_positives = conf_matrix[1, 1]
						false_negatives = conf_matrix[1, 0]
						true_negatives = conf_matrix[0, 0]
						false_positives = conf_matrix[0, 1]
						# Calculate sensitivity (recall)
						sensitivity = true_positives / (true_positives + false_negatives)
						# Calculate selectivity (true negative rate)
						selectivity = true_negatives / (true_negatives + false_positives)
						#print(f"Total Accuracy: {total_accuracy}")
						#print(f"Sensitivity (TP/TP+FN): {sensitivity}")
						#print(f"Selectivity (TN/TN+FP): {selectivity}")
						
						
						emp_TP.append(true_positives)
						emp_FN.append(false_negatives)
						emp_TN.append(true_negatives)
						emp_FP.append(false_positives)
						emp_SEN.append(sensitivity)
						emp_SEL.append(selectivity)
						Lota += 1
						print(Lota)


results_df = pd.DataFrame({
	    'True Positives': emp_TP,
	    'False Negatives': emp_FN,
	    'True Negatives': emp_TN,
	    'False Positives': emp_FP,
	    'Sensitivity': emp_SEN,
	    'Selectivity': emp_SEL
	    })

results_df.to_csv('parameter_matrix_results', sep=' ', index=False)


#def single_predictive():

 #   classifier = joblib.load('image_classifier_model.pkl')
    
    
# Evaluate the accuracy of predicted switches
#accuracy = accuracy_score(df['True_Label'].values[5:], predicted_switches)
#print(f'Accuracy of predicted switches: {accuracy}')
