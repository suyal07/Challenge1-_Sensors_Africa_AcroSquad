#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install numpy


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv(r"File/sensors dataset.csv" , low_memory=False,nrows = 5000000)


# In[4]:


df.tail(1)


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df['parameter'].unique()


# In[8]:


cleaned_df = df.dropna(subset=['parameter'])


# In[9]:


cleaned_df.isnull().sum()


# In[10]:


reduced_df = cleaned_df


# In[11]:


#reduced_df = df.sample(frac=0.1)


# In[12]:


reduced_df.info()


# In[13]:


reduced_df.head(10)


# In[14]:


# Convert 'value' column to numeric
reduced_df['value'] = pd.to_numeric(reduced_df['value'], errors='coerce')

# Pivot the DataFrame to have individual columns for each parameter and values accordingly
pivot_df = reduced_df.pivot_table(index='timestamp', columns='parameter', values='value', fill_value=np.nan).reset_index()


# In[15]:


result_df = pd.merge(reduced_df.drop(columns='value'), pivot_df, on='timestamp').drop(columns='parameter')


# In[16]:


result_df = result_df.drop_duplicates()


# In[17]:


result_df = result_df.sort_values(by='timestamp')


# In[18]:


result_df.head(10)


# In[19]:


result_df.isnull().sum()


# In[20]:


result_df.info()


# In[21]:


result_df['city'].replace(['Nairobi', 'Abuja'],
                        [0, 1], inplace=True)


# In[22]:


result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], utc=True)


# In[24]:


# Handling missing values

result_df['PM 1'].fillna(-1, inplace=True)
result_df['PM 10'].fillna(-1, inplace=True)
result_df['PM 2.5'].fillna(-1, inplace=True)



# In[25]:


# Selecting features and targets
features_columns = ['timestamp', 'device_id', 'sensor_type', 'sensor_id', 'location_id',
                    'city','latitude', 'longitude']

target_columns = ['PM 1', 'PM 10', 'PM 2.5']

# Creating 'features' array excluding the target columns
features = result_df[features_columns].values

# Creating 'targets' array with the target columns
targets = result_df[target_columns].values


# In[26]:


pip install tensorflow 


# In[27]:


#! pip uninstall numpy --yes


# In[28]:


pip install numpy --upgrade --ignore-installed


# In[29]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking


# In[ ]:





# In[30]:


import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))


# In[31]:


# Assuming 'timestamp_column' is the column with Pandas Timestamp objects in your DataFrame
result_df['timestamp'] = pd.to_datetime(result_df['timestamp']).astype(int) // 10**9  # Convert to Unix timestamp (seconds)

# Convert to TensorFlow tensor
result_df['timestamp'] = tf.constant(result_df['timestamp'], dtype=tf.float32)  # Adjust dtype as needed


# In[39]:


result_df.dtypes


# In[40]:


# Assume your data is loaded into 'features' and 'targets' arrays

# Handle missing values in target variables (replace with -1)
targets[np.isnan(targets)] = -1

# Create sequences for LSTM
sequence_length = 30  # Define the length of input sequences

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i : i + sequence_length])
    return np.array(sequences)

# Create input sequences and target sequences
input_sequences = create_sequences(features, sequence_length)
target_sequences = create_sequences(targets, sequence_length)

# Define the LSTM model
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(sequence_length, 8)))  # Masking missing values
model.add(LSTM(units=64, activation='relu'))
model.add(Dense(units=3))  # Adjust units based on the number of target variables

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Split the data into train and test sets
split = int(0.7 * len(input_sequences))
X_train, X_test = input_sequences[:split], input_sequences[split:]
y_train, y_test = target_sequences[:split], target_sequences[split:]

X_train = np.float64(X_train)
X_test = tf.constant(X_test)
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Predict missing values
#predictions = model.predict(X_test)


# In[36]:


X_train.dtype


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Grouping the DataFrame by 'Category'
# grouped = df.groupby(df.timestamp)
# 
# # Splitting the DataFrame into separate DataFrames based on groups
# grouped_df = {group: group_df for group, group_df in grouped}

# # Accessing the separate DataFrames by group name
# for group_name, group_df in grouped_df.items():
#     print(f"Group '{group_name}':\n{group_df}\n")

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




grouped = df.groupby(df.city)
df_abuja = grouped.get_group("Abuja")
df_nairobi = grouped.get_group("Nairobi")grouped_ts = reduced_df.groupby(reduced_df.timestamp)
# In[ ]:




df_sample_1 = grouped_ts.get_group("2023-04-13 12:07:19.663 +0100")df_sample_1df_sample_2 = grouped_ts.get_group("2022-08-06 23:58:54.727 +0100")df_sample_2df_sample_3 = grouped_ts.get_group("2022-09-08 02:15:59.187 +0100")df_sample_3transposed_data = df_sample_3.pivot(index='timestamp', columns='parameter', values='value').reset_index()
transposed_datatransposed_data

# Assuming your data is in a DataFrame named sensor_data

# Extract columns that will remain as is
fixed_columns = ['timestamp', 'device_id', 'chip_id', 'sensor_type', 'sensor_id', 'location_id', 'location',
                 'street_name', 'city', 'country', 'latitude', 'longitude', 'deployment_date']

# Splitting the DataFrame
df_sample_3_data_fixed = df_sample_3[fixed_columns]
df_sample_3_data_values = df_sample_3[['timestamp', 'parameter', 'value']]

# Create a comprehensive set of columns for all parameters
all_parameters = ['Humidity', 'Temperature', 'PM 2.5', 'PM 10', 'PM 1']  # Add other parameters if needed

# Pivot the 'parameter' column into separate columns with corresponding 'value'
transposed_values = df_sample_3_data_values.pivot(index='timestamp', columns='parameter', values='value').reset_index()

# Ensure columns for all parameters exist
for param in all_parameters:
    if param not in transposed_values.columns:
        transposed_values[param] = None

# Sort the columns by timestamp if needed
transposed_values = transposed_values.sort_values('timestamp')

# Merge the transposed values with the fixed columns
transposed_data = pd.merge(df_sample_3_data_fixed.drop_duplicates(subset='timestamp'), transposed_values, on='timestamp')

# Display the transposed data
transposed_data
# Assuming your data is in a DataFrame named sensor_data

# Extract columns that will remain as is
fixed_columns = ['timestamp', 'device_id', 'chip_id', 'sensor_type', 'sensor_id', 'location_id', 'location',
                 'street_name', 'city', 'country', 'latitude', 'longitude', 'deployment_date']

# Splitting the DataFrame
df_fixed = df[fixed_columns]
df_pivot = df[['timestamp', 'parameter', 'value']]

# Create a comprehensive set of columns for all parameters
all_parameters = ['Humidity', 'Temperature', 'PM 2.5', 'PM 10', 'PM 1'] 

# Initialize an empty DataFrame to store the merged results
merged_data = pd.DataFrame(columns=fixed_columns + all_parameters)

# Iterate through unique timestamps
unique_timestamps = df['timestamp'].unique()

for timestamp in unique_timestamps:
    # Filter data for each timestamp
    data_for_timestamp = df_pivot[df_pivot['timestamp'] == timestamp]
    
    # Pivot the 'parameter' column into separate columns with corresponding 'value'
    transposed_values = data_for_timestamp.pivot(index='timestamp', columns='parameter', values='value').reset_index()
    
    # Ensure columns for all parameters exist
    for param in all_parameters:
        if param not in transposed_values.columns:
            transposed_values[param] = None
    
    # Merge the transposed values with the fixed columns
    merged_timestamp_data = pd.merge(df_fixed[df_fixed['timestamp'] == timestamp].drop_duplicates(subset='timestamp'),
                                     transposed_values, on='timestamp')
    
    # Append the merged data for this timestamp to the overall merged_data DataFrame
    merged_data = pd.concat([merged_data, merged_timestamp_data], ignore_index=True)

import pandas as pd

# Assuming your data is in a DataFrame named sensor_data

# Extract columns that will remain as is
fixed_columns = ['timestamp', 'device_id', 'chip_id', 'sensor_type', 'sensor_id', 'location_id', 'location',
                 'street_name', 'city', 'country', 'latitude', 'longitude', 'deployment_date']

# Create a comprehensive set of columns for all parameters
all_parameters = ['Humidity', 'Temperature', 'PM 2.5', 'PM 10', 'PM 1']  # Add other parameters if needed

# Function to pivot and merge data for each timestamp
def process_timestamp(group):
    # Pivot the 'parameter' column into separate columns with corresponding 'value'
    transposed_values = group.pivot(index='timestamp', columns='parameter', values='value').reset_index()
    
    # Ensure columns for all parameters exist
    for param in all_parameters:
        if param not in transposed_values.columns:
            transposed_values[param] = None
    
    # Merge the transposed values with the fixed columns
    merged_timestamp_data = pd.merge(group[fixed_columns].drop_duplicates(subset='timestamp'),
                                     transposed_values, on='timestamp')
    
    return merged_timestamp_data

# Splitting the DataFrame and processing each group
df_fixed_values = df[['timestamp', 'parameter', 'value'] + fixed_columns]
processed_data = df_fixed_values.groupby('timestamp').apply(process_timestamp).reset_index(drop=True)


# Display the processed data for all timestamps
print(processed_data)import pandas as pd

# Assuming your data is in a DataFrame named sensor_data

# Extract columns that will remain as is
fixed_columns = ['timestamp', 'device_id', 'chip_id', 'sensor_type', 'sensor_id', 'location_id', 'location',
                 'street_name', 'city', 'country', 'latitude', 'longitude', 'deployment_date']

# Create a comprehensive set of columns for all parameters
all_parameters = ['Humidity', 'Temperature', 'PM 2.5', 'PM 10', 'PM 1']  # Add other parameters if needed

# Function to pivot and merge data for each timestamp
def process_group(group):
    # Pivot the 'parameter' column into separate columns with corresponding 'value'
    transposed_values = group.pivot(index='timestamp', columns='parameter', values='value').reset_index()
    
    # Ensure columns for all parameters exist
    for param in all_parameters:
        if param not in transposed_values.columns:
            transposed_values[param] = None
    
    # Merge the transposed values with the fixed columns
    merged_timestamp_data = pd.merge(group[fixed_columns].drop_duplicates(subset='timestamp'),
                                     transposed_values, on='timestamp', how='outer')
    
    return merged_timestamp_data

# Splitting the DataFrame and processing each group
df_fixed_values = df[['timestamp', 'parameter', 'value'] + fixed_columns]  # Include fixed columns
grouped = df_fixed_values.groupby(df.timestamp)

# Apply the processing function to each group
processed_data = grouped.apply(process_group).reset_index(drop=True)

# Display the processed data for all timestamps
print(processed_data)

# In[ ]:




