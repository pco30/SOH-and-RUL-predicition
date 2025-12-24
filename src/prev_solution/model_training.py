# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 20:28:33 2025

@author: PRINCELY OSEJI
"""
# Import relevant libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Function to get the creation time of a file
def get_file_creation_time(file_path):
    return os.path.getctime(file_path)

# Function to extract files from a folder path
def extract_files(folder_path):
    excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    return excel_files

# Iterate over each file path and alter the cycle index with addition of more files
def process_files(folder_path, sheet_names_to_import):
    file_paths = extract_files(folder_path)
    file_paths.sort(key=get_file_creation_time)
    dfs = []
    last_cycle_index = 0
    for file_path in file_paths:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        # Process sheets in the order they appear in sheet_names_to_import if they exist in the file
        for target_sheet in sheet_names_to_import:
            if target_sheet in sheet_names:
                df = pd.read_excel(file_path, sheet_name=target_sheet)
                df['Cycle_Index'] += last_cycle_index
                last_cycle_index = df['Cycle_Index'].max()
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# Function to Calculate the State of health of the battery (Constant discharge)
def calculate_soh(df):        
    # Convert 'Test_Time(s)' to hours
    df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
    # Identify the start of each discharge cycle
    discharge_start = (df['Current(A)'] < 0) & (df['Cycle_Index'] == df['Cycle_Index'].shift())
    first_discharge_start_time = df.loc[discharge_start, ['Cycle_Index', 'Test_Time(h)']].groupby('Cycle_Index')['Test_Time(h)'].first()
    first_discharge_start_time_series = df['Cycle_Index'].map(first_discharge_start_time)
    # Calculate test time discharged for each row
    df['Test_Time_Discharged(h)'] = df['Test_Time(h)'] - first_discharge_start_time_series
    # Calculate battery capacity
    df['Battery_Capacity_Per_Point(mAh)'] = df['Test_Time_Discharged(h)'] * abs(df['Current(A)']) * 1000
    df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Battery_Capacity_Per_Point(mAh)'].transform('max')
    max_soh = df['Battery_Capacity(mAh)'].max()
    # Calculate the SOH values by dividing each battery capacity by the maximum value and multiplying by 100
    df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_soh) * 100
    # df['C_rate'] = c_rate
    # Identify the start of each charge cycle
    charge_start = (df['Current(A)'] >= 0) & (df['Cycle_Index'] == df['Cycle_Index'].shift())
    first_charge_start_time = df.loc[charge_start, ['Cycle_Index', 'Test_Time(h)']].groupby('Cycle_Index')['Test_Time(h)'].first()
    first_charge_start_time_series = df['Cycle_Index'].map(first_charge_start_time)
    # Calculate test time charged for each row
    df['Test_Time_charged(h)'] = df['Test_Time(h)'] - first_charge_start_time_series
    return df

# def calculate_soh_variable_discharge(df):
#     # Convert time to hours
#     df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
#     # Sort data just in case
#     df = df.sort_values(['Cycle_Index', 'Test_Time(h)']).reset_index(drop=True)
#     # Compute time difference (delta_t) between consecutive readings
#     df['delta_t(h)'] = df['Test_Time(h)'].diff()
#     df.loc[df['Cycle_Index'] != df['Cycle_Index'].shift(), 'delta_t(h)'] = 0  # Reset at cycle boundaries
#     # Only consider discharge phases: Current < 0
#     df['Discharge_Capacity(mAh)'] = 0.0
#     discharge_mask = df['Current(A)'] < 0
#     df.loc[discharge_mask, 'Discharge_Capacity(mAh)'] = (
#         abs(df.loc[discharge_mask, 'Current(A)']) * df.loc[discharge_mask, 'delta_t(h)'] * 1000
#     )
#     # Sum discharge capacity per cycle
#     df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Discharge_Capacity(mAh)'].transform('sum')
#     df['Battery_Capacity(mAh)'] = df['Battery_Capacity(mAh)']/5 # 5 cycles within each cycle
#     # Calculate SOH
#     max_capacity = df['Battery_Capacity(mAh)'].max()
#     df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_capacity) * 100
#     return df

# Function to calculate soh for variable discharge
def calculate_soh_variable_discharge(df):
    # Convert time to hours
    df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
    # Sort data just in case
    df = df.sort_values(['Cycle_Index', 'Test_Time(h)']).reset_index(drop=True)
    # Compute time difference (delta_t) between consecutive readings
    df['delta_t(h)'] = df['Test_Time(h)'].diff()
    df.loc[df['Cycle_Index'] != df['Cycle_Index'].shift(), 'delta_t(h)'] = 0  # Reset at cycle boundaries
    # Initialize discharge capacity column with float type
    df['Discharge_Capacity(mAh)'] = 0.0
    # Identify discharge rows (Current < 0)
    discharge_mask = df['Current(A)'] < 0
    # Calculate discharge capacity: I × Δt × 1000
    df.loc[discharge_mask, 'Discharge_Capacity(mAh)'] = (
        abs(df.loc[discharge_mask, 'Current(A)']) * df.loc[discharge_mask, 'delta_t(h)'] * 1000
    )
    # Total battery capacity per cycle
    df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Discharge_Capacity(mAh)'].transform('sum')
    df['Battery_Capacity(mAh)'] = df['Battery_Capacity(mAh)']/5 # 5 cycles within each cycle
    # Calculate SOH as a percentage of the maximum capacity
    max_capacity = df['Battery_Capacity(mAh)'].max()
    df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_capacity) * 100
    # Compute charge and discharge times per cycle
    df['Time_charged(h)'] = 0.0
    df['Time_discharged(h)'] = 0.0
    df.loc[df['Current(A)'] >= 0, 'Time_charged(h)'] = df['delta_t(h)']
    df.loc[df['Current(A)'] < 0,  'Time_discharged(h)'] = df['delta_t(h)']
    # Total time charged/discharged per cycle
    df['Test_Time_charged(h)'] = df.groupby('Cycle_Index')['Time_charged(h)'].transform('sum')
    df['Test_Time_Discharged(h)'] = df.groupby('Cycle_Index')['Time_discharged(h)'].transform('sum')
    return df


# Function for selecting valid features
def selection(df):
    # Filter rows where 'Test_Time_Discharged(h)' is greater than 0 and assign 0 to 'Test_Time_charged(h)'
    df.loc[df['Test_Time_Discharged(h)'] > 0, 'Test_Time_charged(h)'] = 0
    
    # Select specific columns
    df = df[['Cycle_Index', 'Internal_Resistance(Ohm)', 'Voltage(V)', 
              'Test_Time_Discharged(h)', 'Test_Time_charged(h)', 'Calculated_SOH(%)', 'Battery_Capacity(mAh)',
              'Charge_Capacity(Ah)', 'Charge_Energy(Wh)', 'dV/dt(V/s)', 'Step_Time(s)']]
    
    # Group by cycle
    cycle_groups = df.groupby('Cycle_Index')

    # Initialize the new column in the original DataFrame
    df['avg_abs_dvdt'] = np.nan
    
    # Iterate through each cycle group
    for cycle, group in cycle_groups:
        # Calculate the average absolute dV/dt for the current cycle
        dvdt = group['dV/dt(V/s)'].dropna()
        if len(dvdt) > 0:
            avg_abs_dvdt_val = np.mean(np.abs(dvdt))

            # Update the 'avg_abs_dvdt' column for rows belonging to the current cycle
            df.loc[df['Cycle_Index'] == cycle, 'avg_abs_dvdt'] = avg_abs_dvdt_val

    # Group the DataFrame by 'Cycle_Index' and apply aggregation functions
    df = df.groupby('Cycle_Index').agg({
        'Internal_Resistance(Ohm)': 'max',
        'Battery_Capacity(mAh)': 'max',
        'Step_Time(s)': 'max',
        'dV/dt(V/s)': 'mean',
        'Charge_Energy(Wh)': 'mean',
        'Charge_Capacity(Ah)': 'mean',
        'Voltage(V)': 'mean',
        'Test_Time_Discharged(h)': 'max',
        'Test_Time_charged(h)': lambda x: x.max() - x.min(), # Corrected code for the calculation
        'Calculated_SOH(%)' : 'max',
        'avg_abs_dvdt': 'max'
    }).reset_index()    
    # Dropping rows with missing or NaN values
    df.dropna(inplace=True)
    return df

# Function for selecting valid features for variable discharge
def selection2(df):
    # Filter rows where 'Test_Time_Discharged(h)' is greater than 0 and assign 0 to 'Test_Time_charged(h)'
    # df.loc[df['Test_Time_Discharged(h)'] > 0, 'Test_Time_charged(h)'] = 0
    
    # Select specific columns
    df = df[['Cycle_Index', 'Internal_Resistance(Ohm)', 'Voltage(V)', 
              'Test_Time_Discharged(h)', 'Test_Time_charged(h)', 'Calculated_SOH(%)', 'Battery_Capacity(mAh)',
              'Charge_Capacity(Ah)', 'Charge_Energy(Wh)', 'dV/dt(V/s)', 'Step_Time(s)']]
    
    # Group by cycle
    cycle_groups = df.groupby('Cycle_Index')

    # Initialize the new column in the original DataFrame
    df['avg_abs_dvdt'] = np.nan
    
    # Iterate through each cycle group
    for cycle, group in cycle_groups:
        # Calculate the average absolute dV/dt for the current cycle
        dvdt = group['dV/dt(V/s)'].dropna()
        if len(dvdt) > 0:
            avg_abs_dvdt_val = np.mean(np.abs(dvdt))

            # Update the 'avg_abs_dvdt' column for rows belonging to the current cycle
            df.loc[df['Cycle_Index'] == cycle, 'avg_abs_dvdt'] = avg_abs_dvdt_val

    # Group the DataFrame by 'Cycle_Index' and apply aggregation functions
    df = df.groupby('Cycle_Index').agg({
        'Internal_Resistance(Ohm)': 'max',
        'Battery_Capacity(mAh)': 'max',
        'Step_Time(s)': 'max',
        'dV/dt(V/s)': 'mean',
        'Charge_Energy(Wh)': 'mean',
        'Charge_Capacity(Ah)': 'mean',
        'Voltage(V)': 'mean',
        'Test_Time_Discharged(h)': 'max',
        'Test_Time_charged(h)': 'max', 
        'Calculated_SOH(%)' : 'max',
        'avg_abs_dvdt': 'max'
    }).reset_index()    
    # Dropping rows with missing or NaN values
    df.dropna(inplace=True)
    return df

# Process the data
datasets = []
folder_paths = ["C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_33",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_34",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_35",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_36",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_37",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_38"]

sheet_names = [["Channel_1-006"], ["Channel_1-007"], ["Channel_1-008"], ["Channel_1-009"], ["Channel_1-010"], ["Channel_1-011"]]

for i in range(len(folder_paths)):
    df = process_files(folder_paths[i], sheet_names[i])
    df = calculate_soh(df)
    df = selection(df)
    df = df[df['Test_Time_charged(h)'] >= 2]
    datasets.append(df)

# Process variable discharge data
folder_paths2 = ["C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_9",
                 "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_3"]

sheet_names2 = [["Channel_1-007", "Channel_1-007_1", "Channel_1-007_2"],
                ["Channel_1-008"]]

for i in range(len(folder_paths2)):
    df2 = process_files(folder_paths2[i], sheet_names2[i])
    df2 = calculate_soh_variable_discharge(df2)
    df2 = selection2(df2)
    # Get SOH values for comparison
    soh = df2['Calculated_SOH(%)']

    # Compute the SOH difference with previous and next rows
    prev_diff = (soh - soh.shift(1)).abs()
    next_diff = (soh - soh.shift(-1)).abs()

    # Identify rows where both diffs are > 3
    mask = (prev_diff > 5) & (next_diff > 5)

    # Drop those rows
    df2 = df2[~mask].reset_index(drop=True)
    datasets.append(df2)

# Concatenate all datasets into one
combined = pd.concat(datasets)
combined = combined[combined['Calculated_SOH(%)'] >= 70] # EOL

# Train Models for prediction
# X = combined.drop(['Calculated_SOH(%)', 'Battery_Capacity(mAh)'], axis=1).values
X = combined.drop(['Calculated_SOH(%)', 'Battery_Capacity(mAh)', 'Step_Time(s)', 'dV/dt(V/s)', 'Charge_Energy(Wh)', 'Charge_Capacity(Ah)', 'avg_abs_dvdt'], axis=1).values
# X = combined[['Cycle_Index', 'Test_Time_Discharged(h)', 'Test_Time_charged(h)']].values
y = combined['Calculated_SOH(%)'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) # Split data

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Normalize y values
y_train = (y_train - 70) / 30
y_test = (y_test - 70) / 30

# Train the neural network (LSTM)
input_shape = X_train_scaled.shape[1]
model = Sequential([
        LSTM(128, input_shape=(input_shape, 1), return_sequences=True),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(1)
    ])

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary() 
model.fit(X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=200, batch_size=50, verbose=1, validation_data=(X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)) 
mse_lstm, mae_lstm = model.evaluate(X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1), y_test)
print("\nLSTM performance")
print("Mean Squared Error:", mse_lstm)
print("Mean Absolute Error:", mae_lstm)

# MULTI-LINEAR REGRESSION
# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate performance
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print("\nLinear Regression Performance")
print("Mean Squared Error:", mse_lr)
print("Mean Absolute Error:", mae_lr)

# XGboost
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=101)

xgb_model.fit(X_train_scaled, y_train.ravel())  # flattens y

# Predict on test data
y_pred = xgb_model.predict(X_test_scaled)

# Evaluate performance
mse_xgb = mean_squared_error(y_test, y_pred)
mae_xgb = mean_absolute_error(y_test, y_pred)

print("\nXGBoost Regressor Performance")
print("Mean Squared Error for XGboost is", mse_xgb)
print("Mean Absolute Error for XGboost is", mae_xgb)

# GRU-RNN
input_shape = X_train.shape[1]
model = Sequential([
        GRU(units=128, return_sequences=True, input_shape=(input_shape, 1)),
        GRU(units=128, return_sequences=False),
        Dense(units=64, activation='relu'),
        Dense(units=1)
    ])

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary() 
model.fit(X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=200, batch_size=50, verbose=1, validation_data=(X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)) 
mse_gru, mae_gru = model.evaluate(X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1), y_test)
print("Mean Squared Error:", mse_gru)
print("Mean Absolute Error:", mae_gru)


# # Define the architecture for the fully connected neural network
# def build_fc_nn(input_shape):
#     model = Sequential()
#     # Input layer is automatically handled by input_shape
#     model.add(Dense(128, input_dim=input_shape, activation='relu'))  # First hidden layer
#     model.add(Dropout(0.2))  # Dropout to prevent overfitting 
#     model.add(Dense(64, activation='relu'))  # Second hidden layer
#     # model.add(Dropout(0.2))  # Dropout layer
#     model.add(Dense(32, activation='relu'))  # Third hidden layer
#     model.add(Dropout(0.2))  # Dropout layer
#     model.add(Dense(16, activation='relu'))  # Fourth hidden layer
#     model.add(Dropout(0.2))  # Dropout layer
#     model.add(Dense(1))  # Output layer (single neuron for regression)
#     # Compile the model
#     optimizer = Adam(learning_rate=0.005)
#     model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
#     return model

# # Build the model with input shape determined from training data
# model_fc = build_fc_nn(X_train_scaled.shape[1])
# # Summarize the model architecture
# model_fc.summary()
# # Train the model
# history_fc = model_fc.fit(X_train_scaled, y_train, epochs=200, batch_size=50, verbose=1, validation_data=(X_test_scaled, y_test))
# # Evaluate the model on the test data
# mse_fc, mae_fc = model_fc.evaluate(X_test_scaled, y_test)

# print("\nFully Connected Neural Network Performance")
# print("Mean Squared Error:", mse_fc)
# print("Mean Absolute Error:", mae_fc)

# ENSEMBLE
# Base model #
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=9, random_state=101)
lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=101)
rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=101)

# OOF function for stacking #
def get_oof_preds(model, X, y, X_test, n_splits=5):
    oof_train = np.zeros((X.shape[0],))
    oof_test = np.zeros((X_test.shape[0],))
    oof_test_skf = np.empty((n_splits, X_test.shape[0]))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        oof_train[val_idx] = model.predict(X_val)
        oof_test_skf[i, :] = model.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

X_train_np = X_train_scaled
X_test_np = X_test_scaled
y_train_np = y_train.ravel()

# Get OOF predictions from base models #
xgb_oof_train, xgb_oof_test = get_oof_preds(xgb, X_train_np, y_train_np, X_test_np)
lgb_oof_train, lgb_oof_test = get_oof_preds(lgb, X_train_np, y_train_np, X_test_np)
rf_oof_train, rf_oof_test = get_oof_preds(rf, X_train_np, y_train_np, X_test_np)

# Stack OOF predictions to create meta features #
X_meta_train = np.concatenate([xgb_oof_train, lgb_oof_train, rf_oof_train], axis=1)
X_meta_test = np.concatenate([xgb_oof_test, lgb_oof_test, rf_oof_test], axis=1)

# Train meta-learner (stacker) #
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train, y_train_np)
y_pred_stack = meta_model.predict(X_meta_test)

# Evaluate model
mse_stack = mean_squared_error(y_test, y_pred_stack)
mae_stack = mean_absolute_error(y_test, y_pred_stack)

print("Mean Squared Error:", round(mse_stack, 6))
print("Mean Absolute Error:", round(mae_stack, 6))


# # -*- coding: utf-8 -*-
# """
# Created on Fri Apr 11 20:28:33 2025

# @author: PRINCELY OSEJI
# """
# # import relevant libraries
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
# from keras.models import Sequential
# from keras.optimizers import Adam
# from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout
# from sklearn.ensemble import RandomForestRegressor
# from lightgbm import LGBMRegressor
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import KFold

# # Function to get the creation time of a file
# def get_file_creation_time(file_path):
#     return os.path.getctime(file_path)

# # Function to extract files from a folder path
# def extract_files(folder_path):
#     excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
#     return excel_files

# # Iterate over each file path and alter the cycle index with addition of more files
# def process_files(folder_path, sheet_names_to_import):
#     file_paths = extract_files(folder_path)
#     file_paths.sort(key=get_file_creation_time)
#     dfs = []
#     last_cycle_index = 0
#     for file_path in file_paths:
#         sheet_names = pd.ExcelFile(file_path).sheet_names
#         # Process sheets in the order they appear in sheet_names_to_import if they exist in the file
#         for target_sheet in sheet_names_to_import:
#             if target_sheet in sheet_names:
#                 df = pd.read_excel(file_path, sheet_name=target_sheet)
#                 df['Cycle_Index'] += last_cycle_index
#                 last_cycle_index = df['Cycle_Index'].max()
#                 dfs.append(df)
#     return pd.concat(dfs, ignore_index=True)


# # Function to Calculate the State of health of the battery (Constant discharge)
# def calculate_soh(df):        
#     # Convert 'Test_Time(s)' to hours
#     df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
#     # Identify the start of each discharge cycle
#     discharge_start = (df['Current(A)'] < 0) & (df['Cycle_Index'] == df['Cycle_Index'].shift())
#     first_discharge_start_time = df.loc[discharge_start, ['Cycle_Index', 'Test_Time(h)']].groupby('Cycle_Index')['Test_Time(h)'].first()
#     first_discharge_start_time_series = df['Cycle_Index'].map(first_discharge_start_time)
#     # Calculate test time discharged for each row
#     df['Test_Time_Discharged(h)'] = df['Test_Time(h)'] - first_discharge_start_time_series
#     # Calculate battery capacity
#     df['Battery_Capacity_Per_Point(mAh)'] = df['Test_Time_Discharged(h)'] * abs(df['Current(A)']) * 1000
#     df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Battery_Capacity_Per_Point(mAh)'].transform('max')
#     max_soh = df['Battery_Capacity(mAh)'].max()
#     # Calculate the SOH values by dividing each battery capacity by the maximum value and multiplying by 100
#     df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_soh) * 100
#     # df['C_rate'] = c_rate
#     # Identify the start of each charge cycle
#     charge_start = (df['Current(A)'] >= 0) & (df['Cycle_Index'] == df['Cycle_Index'].shift())
#     first_charge_start_time = df.loc[charge_start, ['Cycle_Index', 'Test_Time(h)']].groupby('Cycle_Index')['Test_Time(h)'].first()
#     first_charge_start_time_series = df['Cycle_Index'].map(first_charge_start_time)
#     # Calculate test time charged for each row
#     df['Test_Time_charged(h)'] = df['Test_Time(h)'] - first_charge_start_time_series
    
#     # Compute mean voltage for charge and discharge per cycle and assign
#     mean_voltage_charge = (
#         df[df['Current(A)'] >= 0]
#         .groupby('Cycle_Index')['Voltage(V)']
#         .mean()
#         .rename('mean_voltage_charge')
#     )
#     mean_voltage_discharge = (
#         df[df['Current(A)'] < 0]
#         .groupby('Cycle_Index')['Voltage(V)']
#         .mean()
#         .rename('mean_voltage_discharge')
#     )
#     df['mean_voltage_charge'] = df['Cycle_Index'].map(mean_voltage_charge)
#     df['mean_voltage_discharge'] = df['Cycle_Index'].map(mean_voltage_discharge)
#     return df

# # Function to calculate soh for variable discharge
# def calculate_soh_variable_discharge(df):
#     # Convert time to hours
#     df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
#     # Sort data just in case
#     df = df.sort_values(['Cycle_Index', 'Test_Time(h)']).reset_index(drop=True)
#     # Compute time difference (delta_t) between consecutive readings
#     df['delta_t(h)'] = df['Test_Time(h)'].diff()
#     df.loc[df['Cycle_Index'] != df['Cycle_Index'].shift(), 'delta_t(h)'] = 0  # Reset at cycle boundaries
#     # Initialize discharge capacity column with float type
#     df['Discharge_Capacity(mAh)'] = 0.0
#     # Identify discharge rows (Current < 0)
#     discharge_mask = df['Current(A)'] < 0
#     # Calculate discharge capacity: I × Δt × 1000
#     df.loc[discharge_mask, 'Discharge_Capacity(mAh)'] = (
#         abs(df.loc[discharge_mask, 'Current(A)']) * df.loc[discharge_mask, 'delta_t(h)'] * 1000
#     )
#     # Total battery capacity per cycle
#     df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Discharge_Capacity(mAh)'].transform('sum')
#     df['Battery_Capacity(mAh)'] = df['Battery_Capacity(mAh)']/5 # 5 cycles within each cycle
#     # Calculate SOH as a percentage of the maximum capacity
#     max_capacity = df['Battery_Capacity(mAh)'].max()
#     df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_capacity) * 100
#     # Compute charge and discharge times per cycle
#     df['Time_charged(h)'] = 0.0
#     df['Time_discharged(h)'] = 0.0
#     df.loc[df['Current(A)'] >= 0, 'Time_charged(h)'] = df['delta_t(h)']
#     df.loc[df['Current(A)'] < 0,  'Time_discharged(h)'] = df['delta_t(h)']
#     # Total time charged/discharged per cycle
#     df['Test_Time_charged(h)'] = df.groupby('Cycle_Index')['Time_charged(h)'].transform('sum')
#     df['Test_Time_Discharged(h)'] = df.groupby('Cycle_Index')['Time_discharged(h)'].transform('sum')
    
#     # Compute mean voltage for charge and discharge per cycle and assign
#     mean_voltage_charge = (
#         df[df['Test_Time_charged(h)'] > 0]
#         .groupby('Cycle_Index')['Voltage(V)']
#         .mean()
#         .rename('mean_voltage_charge')
#     )
#     mean_voltage_discharge = (
#         df[df['Time_discharged(h)'] > 0]
#         .groupby('Cycle_Index')['Voltage(V)']
#         .mean()
#         .rename('mean_voltage_discharge')
#     )
#     df['mean_voltage_charge'] = df['Cycle_Index'].map(mean_voltage_charge)
#     df['mean_voltage_discharge'] = df['Cycle_Index'].map(mean_voltage_discharge)
#     return df


# # Function for selecting valid features
# def selection(df):
#     # Filter rows where 'Test_Time_Discharged(h)' is greater than 0 and assign 0 to 'Test_Time_charged(h)'
#     df.loc[df['Test_Time_Discharged(h)'] > 0, 'Test_Time_charged(h)'] = 0
    
#     # Select specific columns
#     df = df[['Cycle_Index', 'Internal_Resistance(Ohm)', 'Voltage(V)', 
#               'Test_Time_Discharged(h)', 'Test_Time_charged(h)', 'Calculated_SOH(%)', 'Battery_Capacity(mAh)',
#               'Charge_Capacity(Ah)', 'Charge_Energy(Wh)', 'dV/dt(V/s)', 'Step_Time(s)', 'mean_voltage_charge', 'mean_voltage_discharge']]
    
#     # Group by cycle
#     cycle_groups = df.groupby('Cycle_Index')

#     # Initialize the new column in the original DataFrame
#     df['avg_abs_dvdt'] = np.nan
    
#     # Iterate through each cycle group
#     for cycle, group in cycle_groups:
#         # Calculate the average absolute dV/dt for the current cycle
#         dvdt = group['dV/dt(V/s)'].dropna()
#         if len(dvdt) > 0:
#             avg_abs_dvdt_val = np.mean(np.abs(dvdt))

#             # Update the 'avg_abs_dvdt' column for rows belonging to the current cycle
#             df.loc[df['Cycle_Index'] == cycle, 'avg_abs_dvdt'] = avg_abs_dvdt_val

#     # Group the DataFrame by 'Cycle_Index' and apply aggregation functions
#     df = df.groupby('Cycle_Index').agg({
#         'Internal_Resistance(Ohm)': 'max',
#         'Battery_Capacity(mAh)': 'max',
#         'Step_Time(s)': 'max',
#         'dV/dt(V/s)': 'mean',
#         'Charge_Energy(Wh)': 'mean',
#         'Charge_Capacity(Ah)': 'mean',
#         'Voltage(V)': 'mean',
#         'Test_Time_Discharged(h)': 'max',
#         'Test_Time_charged(h)': lambda x: x.max() - x.min(), # Corrected code for the calculation
#         'Calculated_SOH(%)' : 'max',
#         'avg_abs_dvdt': 'max',
#         'mean_voltage_charge': 'max',
#         'mean_voltage_discharge': 'max'
#     }).reset_index()    
#     # Dropping rows with missing or NaN values
#     df['resistance_diff'] = df['Internal_Resistance(Ohm)'].diff()
#     df.dropna(inplace=True)
#     return df

# # Function for selecting valid features for variable discharge
# def selection2(df):
#     # Filter rows where 'Test_Time_Discharged(h)' is greater than 0 and assign 0 to 'Test_Time_charged(h)'
#     # df.loc[df['Test_Time_Discharged(h)'] > 0, 'Test_Time_charged(h)'] = 0
    
#     # Select specific columns
#     df = df[['Cycle_Index', 'Internal_Resistance(Ohm)', 'Voltage(V)', 
#               'Test_Time_Discharged(h)', 'Test_Time_charged(h)', 'Calculated_SOH(%)', 'Battery_Capacity(mAh)',
#               'Charge_Capacity(Ah)', 'Charge_Energy(Wh)', 'dV/dt(V/s)', 'Step_Time(s)', 'mean_voltage_charge', 'mean_voltage_discharge']]
    
#     # Group by cycle
#     cycle_groups = df.groupby('Cycle_Index')

#     # Initialize the new column in the original DataFrame
#     df['avg_abs_dvdt'] = np.nan
    
#     # Iterate through each cycle group
#     for cycle, group in cycle_groups:
#         # Calculate the average absolute dV/dt for the current cycle
#         dvdt = group['dV/dt(V/s)'].dropna()
#         if len(dvdt) > 0:
#             avg_abs_dvdt_val = np.mean(np.abs(dvdt))

#             # Update the 'avg_abs_dvdt' column for rows belonging to the current cycle
#             df.loc[df['Cycle_Index'] == cycle, 'avg_abs_dvdt'] = avg_abs_dvdt_val

#     # Group the DataFrame by 'Cycle_Index' and apply aggregation functions
#     df = df.groupby('Cycle_Index').agg({
#         'Internal_Resistance(Ohm)': 'max',
#         'Battery_Capacity(mAh)': 'max',
#         'Step_Time(s)': 'max',
#         'dV/dt(V/s)': 'mean',
#         'Charge_Energy(Wh)': 'mean',
#         'Charge_Capacity(Ah)': 'mean',
#         'Voltage(V)': 'mean',
#         'Test_Time_Discharged(h)': 'max',
#         'Test_Time_charged(h)': 'max', 
#         'Calculated_SOH(%)' : 'max',
#         'avg_abs_dvdt': 'max',
#         'mean_voltage_charge': 'max',
#         'mean_voltage_discharge': 'max'
#     }).reset_index()    
#     # Dropping rows with missing or NaN values
#     df['resistance_diff'] = df['Internal_Resistance(Ohm)'].diff()
#     df.dropna(inplace=True)
#     return df

# # Process the data
# datasets = []
# folder_paths = ["C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_33",
#                 "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_34",
#                 "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_35",
#                 "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_36",
#                 "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_37",
#                 "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_38"]

# sheet_names = [["Channel_1-006"], ["Channel_1-007"], ["Channel_1-008"], ["Channel_1-009"], ["Channel_1-010"], ["Channel_1-011"]]

# for i in range(len(folder_paths)):
#     df = process_files(folder_paths[i], sheet_names[i])
#     df = calculate_soh(df)
#     df = selection(df)
#     df = df[df['Test_Time_charged(h)'] >= 2]
#     datasets.append(df)

# # Process variable discharge data
# folder_paths2 = ["C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_9",
#                  "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_3"]

# sheet_names2 = [["Channel_1-007", "Channel_1-007_1", "Channel_1-007_2"],
#                 ["Channel_1-008"]]

# for i in range(len(folder_paths2)):
#     df2 = process_files(folder_paths2[i], sheet_names2[i])
#     df2 = calculate_soh_variable_discharge(df2)
#     df2 = selection2(df2)
#     # Get SOH values for comparison
#     soh = df2['Calculated_SOH(%)']

#     # Compute the SOH difference with previous and next rows
#     prev_diff = (soh - soh.shift(1)).abs()
#     next_diff = (soh - soh.shift(-1)).abs()

#     # Identify rows where both diffs are > 3
#     mask = (prev_diff > 5) & (next_diff > 5)

#     # Drop those rows
#     df2 = df2[~mask].reset_index(drop=True)
#     datasets.append(df2)

# # Concatenate all datasets into one
# combined = pd.concat(datasets)
# combined = combined[combined['Calculated_SOH(%)'] >= 70] # EOL

# # Train Models for prediction
# # X = combined.drop(['Calculated_SOH(%)', 'Battery_Capacity(mAh)'], axis=1).values
# X = combined.drop(['Calculated_SOH(%)', 'Battery_Capacity(mAh)', 'Step_Time(s)', 'dV/dt(V/s)', 'Charge_Energy(Wh)', 'Charge_Capacity(Ah)'], axis=1).values
# # X = combined[['Cycle_Index', 'Test_Time_Discharged(h)', 'Test_Time_charged(h)']].values
# y = combined['Calculated_SOH(%)'].values.reshape(-1, 1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Train the neural network (LSTM)
# input_shape = X_train_scaled.shape[1]
# model = Sequential([
#         LSTM(128, input_shape=(input_shape, 1), return_sequences=True),
#         LSTM(128, return_sequences=False, activation='relu'),
#         Dense(1)
#     ])

# optimizer = Adam(learning_rate=0.01)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
# model.summary() 
# model.fit(X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=200, batch_size=50, verbose=1, validation_data=(X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)) 
# mse_lstm, mae_lstm = model.evaluate(X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1), y_test)
# print("\nLSTM performance")
# print("Mean Squared Error:", mse_lstm)
# print("Mean Absolute Error:", mae_lstm)

# # MULTI-LINEAR REGRESSION
# # Initialize and train the model
# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train)

# # Predict on test data
# y_pred_lr = lr_model.predict(X_test_scaled)

# # Evaluate performance
# mse_lr = mean_squared_error(y_test, y_pred_lr)
# mae_lr = mean_absolute_error(y_test, y_pred_lr)

# print("\nLinear Regression Performance")
# print("Mean Squared Error:", mse_lr)
# print("Mean Absolute Error:", mae_lr)

# # XGboost
# xgb_model = XGBRegressor(
#     n_estimators=200,
#     learning_rate=0.1,
#     max_depth=5,
#     random_state=101)

# xgb_model.fit(X_train_scaled, y_train.ravel())  # flattens y

# # Predict on test data
# y_pred = xgb_model.predict(X_test_scaled)

# # Evaluate performance
# mse_xgb = mean_squared_error(y_test, y_pred)
# mae_xgb = mean_absolute_error(y_test, y_pred)

# print("\nXGBoost Regressor Performance")
# print("Mean Squared Error for XGboost is", mse_xgb)
# print("Mean Absolute Error for XGboost is", mae_xgb)

# # GRU-RNN
# input_shape = X_train.shape[1]
# model = Sequential([
#         GRU(units=128, return_sequences=True, input_shape=(input_shape, 1)),
#         GRU(units=128, return_sequences=False),
#         Dense(units=64, activation='relu'),
#         Dense(units=1)
#     ])

# optimizer = Adam(learning_rate=0.01)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
# model.summary() 
# model.fit(X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=200, batch_size=50, verbose=1, validation_data=(X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)) 
# mse_gru, mae_gru = model.evaluate(X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1), y_test)
# print("Mean Squared Error:", mse_gru)
# print("Mean Absolute Error:", mae_gru)


# # # Define the architecture for the fully connected neural network
# # def build_fc_nn(input_shape):
# #     model = Sequential()
# #     # Input layer is automatically handled by input_shape
# #     model.add(Dense(128, input_dim=input_shape, activation='relu'))  # First hidden layer
# #     model.add(Dropout(0.2))  # Dropout to prevent overfitting 
# #     model.add(Dense(64, activation='relu'))  # Second hidden layer
# #     # model.add(Dropout(0.2))  # Dropout layer
# #     model.add(Dense(32, activation='relu'))  # Third hidden layer
# #     model.add(Dropout(0.2))  # Dropout layer
# #     model.add(Dense(16, activation='relu'))  # Fourth hidden layer
# #     model.add(Dropout(0.2))  # Dropout layer
# #     model.add(Dense(1))  # Output layer (single neuron for regression)
# #     # Compile the model
# #     optimizer = Adam(learning_rate=0.005)
# #     model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
# #     return model

# # # Build the model with input shape determined from training data
# # model_fc = build_fc_nn(X_train_scaled.shape[1])
# # # Summarize the model architecture
# # model_fc.summary()
# # # Train the model
# # history_fc = model_fc.fit(X_train_scaled, y_train, epochs=200, batch_size=50, verbose=1, validation_data=(X_test_scaled, y_test))
# # # Evaluate the model on the test data
# # mse_fc, mae_fc = model_fc.evaluate(X_test_scaled, y_test)

# # print("\nFully Connected Neural Network Performance")
# # print("Mean Squared Error:", mse_fc)
# # print("Mean Absolute Error:", mae_fc)

# # ENSEMBLE
# # Base model #
# xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=9, random_state=101)
# lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=101)
# rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=101)

# # OOF function for stacking #
# def get_oof_preds(model, X, y, X_test, n_splits=5):
#     oof_train = np.zeros((X.shape[0],))
#     oof_test = np.zeros((X_test.shape[0],))
#     oof_test_skf = np.empty((n_splits, X_test.shape[0]))

#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     for i, (train_idx, val_idx) in enumerate(kf.split(X)):
#         X_tr, X_val = X[train_idx], X[val_idx]
#         y_tr, y_val = y[train_idx], y[val_idx]

#         model.fit(X_tr, y_tr)
#         oof_train[val_idx] = model.predict(X_val)
#         oof_test_skf[i, :] = model.predict(X_test)

#     oof_test[:] = oof_test_skf.mean(axis=0)
#     return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# X_train_np = X_train_scaled
# X_test_np = X_test_scaled
# y_train_np = y_train.ravel()

# # Get OOF predictions from base models #
# xgb_oof_train, xgb_oof_test = get_oof_preds(xgb, X_train_np, y_train_np, X_test_np)
# lgb_oof_train, lgb_oof_test = get_oof_preds(lgb, X_train_np, y_train_np, X_test_np)
# rf_oof_train, rf_oof_test = get_oof_preds(rf, X_train_np, y_train_np, X_test_np)

# # Stack OOF predictions to create meta features #
# X_meta_train = np.concatenate([xgb_oof_train, lgb_oof_train, rf_oof_train], axis=1)
# X_meta_test = np.concatenate([xgb_oof_test, lgb_oof_test, rf_oof_test], axis=1)

# # Train meta-learner (stacker) #
# meta_model = Ridge(alpha=1.0)
# meta_model.fit(X_meta_train, y_train_np)
# y_pred_stack = meta_model.predict(X_meta_test)

# # Evaluate model
# mse_stack = mean_squared_error(y_test, y_pred_stack)
# mae_stack = mean_absolute_error(y_test, y_pred_stack)

# print("Mean Squared Error:", round(mse_stack, 6))
# print("Mean Absolute Error:", round(mae_stack, 6))

