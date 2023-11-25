import pandas as pd
import numpy as np
def add_timestamp_column(df):

    timestamp = []
    truelabel = np.array(df['True_Label'])
    print(len(truelabel))
    mem = truelabel[0]
    #print(initmem)
    counter = 0
    for value in truelabel:
        if value == mem:
    	    counter += 1
        else:
            counter = 1
            mem = value
        timestamp.append(counter)        
    
    timestamp = np.array(timestamp)
    df['Reset_Timestamps'] = timestamp
    return df

# Example usage:
# Assuming df is your DataFrame with 'true label' column
# df = pd.DataFrame({'true label': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]})
# df_with_timestamp = add_timestamp_column(df)
# print(df_with_timestamp)



prediction_results = pd.read_csv('prediction_results_sorted', delimiter=' ')
prediction_results = add_timestamp_column(prediction_results)
prediction_results.to_csv('prediction_results_sorted_resetstamp', sep=' ', index=False)
