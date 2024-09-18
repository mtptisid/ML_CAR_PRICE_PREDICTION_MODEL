import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Set the style for the plots
mpl.style.use('ggplot')

# Load the data
car = pd.read_csv('quikr_car.csv')

# Print the first few rows of the data
print(car.head())

# Get the shape of the data
print(car.shape)

# Get the info of the data
print(car.info())

# Create a backup copy of the data
backup = car.copy()

# Clean the data
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]

# Print the shape of the cleaned data
print(car.shape)

# Print the info of the cleaned data
print(car.info())

# Reset the index of the cleaned data
car = car.reset_index(drop=True)

# Print the first few rows of the cleaned data
print(car.head())

# Save the cleaned data to a new CSV file
car.to_csv('Cleaned_Car_data.csv')

# Print the unique values of the 'company' column
print(car['company'].unique())

# Create a plot of the relationship between 'company' and 'Price'
plt.figure(figsize=(10, 6))
sns.boxplot(x='company', y='Price', data=car)
plt.title('Relationship between Company and Price')
plt.show()

# Print the summary statistics of the data
print(car.describe(include='all'))

# Filter the data to only include rows where the price is less than 6 million
car = car[car['Price'] < 6000000]

# Print the unique values of the 'company' column
print(car['company'].unique())

# Create a plot of the relationship between 'company' and 'Price'
plt.figure(figsize=(10, 6))
sns.boxplot(x='company', y='Price', data=car)
plt.title('Relationship between Company and Price')
plt.show()

# Print the summary statistics of the data
print(car.describe(include='all'))

# Split the data into training and testing sets
X = car[['year', 'kms_driven']]
y = car['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Create a scatter plot of the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()

# Save the model to a file using joblib
with open('LinearRegressionModel.pkl', 'wb') as f:
    joblib.dump(model, f)