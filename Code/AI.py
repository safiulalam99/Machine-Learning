import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
# read data

df = pd.read_csv(r'C:\Users\saifs\Downloads\startup.csv')
new = pd.read_csv(r'C:\Users\saifs\Downloads\new_company.csv')

y = df.iloc[:,[3]] # Profit
X = df.iloc[:,[0,1,2]] # R&D Spend,  Administration , Marketing Spend

new_X = new.iloc[:,[0,1,2]] # R&D Spend,  Administration , Marketing Spend
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# train the model
model = LinearRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print ('\nTestdata metrics:')
print (f'mae: {mae}')
print (f'mse: {mse}')
print (f'rmse: {rmse}')
print(f'R2: {r2}') 


print (f'Prediction of profit of a neww company: {np.round(model.predict(new_X),2)}\n' )
