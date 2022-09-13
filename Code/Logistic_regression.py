import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

### Prepare data ###

# Read data to pandas dataframe
df = pd.read_csv(r'C:\Users\saifs\OneDrive\Desktop\TAMK\SEM5\AI\dataSet\sales_data.csv',sep=';',quotechar='"')

# Create input data (X) and output data (y)
X = df[['Weekday','Seller']] 
y = df[['Sales Rating']]

# Split data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save the original format data in readable format for the later testing phase
X_test_orig = X_test

# Transform categorical text values in two columns into numeric columns for each category in column
# pandas.get_dummies transforms only categorical columns, and it does not touch numerical columns
# Use drop_first to remove one column to avoid the 'dummy variable trap'
y_train = pd.get_dummies(X_train, drop_first=True)
y_test = pd.get_dummies(X_test, drop_first=True)

# Feature scaling - normalise (scale) all numeric data into range [0, 1]
# In this case column 'Age' must be scale to [0, 1] to prevent 'Age' column domination in the model


# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)