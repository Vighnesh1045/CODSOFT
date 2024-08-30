import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df =pd.read_csv('movies.csv',encoding='latin1')
df.head(10)

df.shape

df.info()

df.columns

# Droping columns
df = df.drop(columns=['Name', 'Actor 2', 'Actor 3'])
df.head(10)

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)
df.shape


# data transformation
df['Year'].unique()

def years(value):
  value = str(value).strip('()')
  return int(value)
df['Year'] = df['Year'].apply(years)
df['Year'].head()

df['Duration'].unique()

def durations(value):
  value = str(value).split(' ')
  value = value[0]
  return int(value)
df['Duration'] = df['Duration'].apply(durations)
df['Duration'].head()

df['Genre'].unique()

def genres(df,Genre):

    df['Genre1'] = df[Genre].str.split(',', expand=True)[0]
    df['Genre2'] = df[Genre].str.split(',', expand=True)[1]
    df['Genre3'] = df[Genre].str.split(',', expand=True)[2]
    return df
df = genres(df,'Genre')
df.head()

df.isna().sum()

df = df.fillna(0)
df.isna().sum()

G=['Genre1','Genre2','Genre3']
for x in G:
    df[x],_ = pd.factorize(df[x])

df = df.drop(columns=['Genre'])
df.head(3)

df.columns

df['Votes'].unique()

def handleVotes(value):
    value = str(value).replace(',','')
    return int(value)
df['Votes'] = df['Votes'].apply(handleVotes)
df['Votes'].head()

# plottind distribution of movie ratings
plt.figure(figsize=(10,6))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

''' There is error indicates that there are still some non-numeric values
 in  DataFrame, which are causing issues when trying to compute
 the correlation matrix. '''

# Convert non-numeric columns before computing the correlation
df_numeric = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
X = df.drop(columns=['Rating'])
y = df['Rating']
categorical_columns = ['Genre1', 'Genre2', 'Genre3']
X_encoded = pd.get_dummies(X, columns=categorical_columns)
model = RandomForestRegressor()
model.fit(X_encoded, y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.title("Feature Importance")
plt.bar(range(X_encoded.shape[1]), importances[indices], align="center")
plt.xticks(range(X_encoded.shape[1]), X_encoded.columns[indices], rotation=90)
plt.xlim([-1, X_encoded.shape[1]])
plt.show()