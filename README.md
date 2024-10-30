Weight Change Prediction Model

This project is a neural network model built to predict weight changes based on various factors like age, gender, daily caloric intake, physical activity level, and more. It uses a dataset containing various health and activity metrics to estimate weight fluctuations over a specified period.
Table of Contents

    Dataset
    Installation
    Data Preprocessing
    Model Architecture
    Training the Model
    Saving the Model

Dataset

The dataset used for this project, weight_change_dataset.csv, contains columns such as:

    Participant ID
    Age
    Gender
    Current Weight (lbs)
    BMR (Calories)
    Daily Calories Consumed
    Daily Caloric Surplus/Deficit
    Weight Change (lbs)
    Duration (weeks)
    Physical Activity Level
    Sleep Quality
    Stress Level
    Final Weight (lbs)

These columns provide insights into each participant's weight metrics and lifestyle, enabling the model to make predictions based on specific health patterns.
Installation

To run this project, ensure you have the following libraries installed:

bash

pip install pandas tensorflow scikit-learn

Data Preprocessing
Step 1: Load the Dataset

Load the dataset from the specified file path and inspect the first few rows:

python

import pandas as pd

# Load the dataset
path = "/path/to/weight_change_dataset.csv"
dataset = pd.read_csv(path)
print(dataset.head())

Step 2: Handle Missing Values

Remove or impute any missing values in the dataset to maintain data integrity:

python

dataset.dropna(inplace=True)

Step 3: Encode Categorical Variables

Convert categorical features (like Gender, Physical Activity Level, Sleep Quality) into numeric form using one-hot encoding:

python

dataset = pd.get_dummies(dataset, columns=['Gender', 'Physical Activity Level', 'Sleep Quality'], drop_first=True)

Step 4: Normalize Continuous Variables

Standardize numerical columns to improve model performance:

python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_columns = ['Age', 'Current Weight (lbs)', 'BMR (Calories)', 'Daily Calories Consumed', 
                   'Daily Caloric Surplus/Deficit', 'Duration (weeks)', 'Stress Level']
dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

Step 5: Split the Dataset

Divide the dataset into features (X) and target (y) for training:

python

X = dataset.drop(columns=['Weight Change (lbs)'])
y = dataset['Weight Change (lbs)']

Model Architecture

The model uses a feedforward neural network with the following structure:

    Input layer matching the feature size
    Two hidden layers with ReLU activation
    Output layer with a single neuron for predicting weight change

python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Single output for regression
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

Training the Model

Train the model on 80% of the data and validate it on the remaining 20%:

python

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

Saving the Model

After training, save the model for future use:

python

model.save('weight_change_model.h5')
