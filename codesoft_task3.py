# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# loding dataset
df = pd.read_csv('IRIS.csv')

# understanding data
df.head(150)

df.info()

df.shape

df.isnull().sum()

df.describe()

df['species'].value_counts()

# visualizing

sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plot.show()

plot.figure(figsize=(12, 6))

plot.subplot(2, 2, 1)
sns.boxplot(x='species', y='sepal_length', data=df)
plot.title('Sepal Length vs Species')

plot.subplot(2, 2, 2)
sns.boxplot(x='species', y='sepal_width', data=df)
plot.title('Sepal Width vs Species')

plot.subplot(2, 2, 3)
sns.boxplot(x='species', y='petal_length', data=df)
plot.title('Petal Length vs Species')

plot.subplot(2, 2, 4)
sns.boxplot(x='species', y='petal_width', data=df)
plot.title('Petal Width vs Species')

plot.tight_layout()
plot.show()

# Jointplot for Sepal Length and Sepal Width
sns.jointplot(x='sepal_length', y='sepal_width', data=df, hue='species', kind='scatter')
plot.show()

# Jointplot for Petal Length and Petal Width
sns.jointplot(x='petal_length', y='petal_width', data=df, hue='species', kind='scatter')
plot.show()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


print("Classification Report:\n", classification_report(y_test, y_pred))


# saving the model for future use
import joblib
joblib.dump(model, 'iris_model.pkl')
loaded_model = joblib.load('iris_model.pkl')


print("\nInput the sepal and petal measurements:")

sepal_length = float(input("Enter sepal length (in cm): "))
sepal_width = float(input("Enter sepal width (in cm): "))
petal_length = float(input("Enter petal length (in cm): "))
petal_width = float(input("Enter petal width (in cm): "))

# Prepare the i/p data
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Predict the species
prediction = loaded_model.predict(input_data)
predicted_species = label_encoder.inverse_transform(prediction)


print(f"\nThe predicted species is: {predicted_species[0]}")