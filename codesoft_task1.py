# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loding dataset
df = pd.read_csv('Titanic-Dataset.csv')
df.head()

df['Survived'].value_counts()

# visulizing
sns.countplot(x='Survived', data=df)

sns.countplot(x='Survived', hue='Sex', data=df)

sns.countplot(x='Survived', hue='Pclass', data=df)

sns.countplot(x='Pclass', hue='Sex', data=df)

# tranforming
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] =le.fit_transform(df['Sex'])
df.head()

df.describe()

df.isna().sum()

df.drop('Age', axis=1, inplace=True)

print(df.columns)

df.drop(['Cabin','PassengerId','Name','SibSp','Parch','Embarked'], axis=1)

X = df[['Pclass','Sex']]
Y = df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
model.fit(X_train, Y_train)

prediction = print(model.predict(X_test))

print(Y_test)

import warnings
warnings.filterwarnings('ignore')


result = model.predict([[1,1]])

if(result==0):
    print("Dead")
else:
    print("Alive")