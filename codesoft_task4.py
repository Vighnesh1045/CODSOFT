# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# loading dataset
df = pd.read_csv('advertising.csv')

# understading data
df.head()

df.info()

df.shape

df.describe()

df.isnull().sum()

#TV vs Sales

sns.scatterplot(x='TV', y='Sales', data=df)
plt.title('TV Advertising vs Sales')
plt.xlabel('TV Advertising Spend')
plt.ylabel('Sales')
plt.show()


#Radio vs Sales

sns.scatterplot(x='Radio', y='Sales', data=df)
plt.title('Radio Advertising vs Sales')
plt.xlabel('Radio Advertising Spend')
plt.ylabel('Sales')
plt.show()


#Newspaper vs Sales

sns.scatterplot(x='Newspaper', y='Sales', data=df)
plt.title('Newspaper Advertising vs Sales')
plt.xlabel('Newspaper Advertising Spend')
plt.ylabel('Sales')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to train and evaluate the model for a given feature
def evaluate_feature(feature):
    X = df[[feature]]
    y = df['Sales']

    # Split the data into 80/20 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = LinearRegression()
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error and R-squared
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Feature: {feature}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}\n')

    # Interpret the coefficients
    coef = model.coef_[0]
    intercept = model.intercept_
    print(f'Equation: Sales = {intercept:.2f} + {coef:.2f} * {feature}')

    # Coefficient (slope) and Intercept
    print(f'Coefficient (Slope): {model.coef_[0]}')
    print(f'Intercept: {model.intercept_}')


    # Visualizing
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted Line')
    plt.xlabel(feature)
    plt.ylabel('Sales')
    plt.title(f'Actual vs. Predicted Sales for {feature}')
    plt.legend()
    plt.show()

# testing individually
for feature in ['TV', 'Radio', 'Newspaper']:
    evaluate_feature(feature)

new_data = [[63.5, 36.4, 78.2]]
predicted_sales = model.predict(new_data)
print(f'Predicted Sales for TV, RADIO AND NEWSPAPER SPEND: {predicted_sales[0]}')