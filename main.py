# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')

plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True, cmap='terrain')

sns.pairplot(data=df)

df.hist(figsize=(12, 12), layout=(5, 3))

df.plot(kind='box', subplots=True, layout=(5, 3), figsize=(12, 12))


sns.catplot(data=df, x='sex', y='age', hue='target', palette='husl')

sns.barplot(data=df, x='sex', y='chol', hue='target', palette='spring')

StandardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])

X= df.drop(['target'], axis=1)
y= df['target']
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=40)
print('X_train-', X_train.size)
print('X_test-', X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)
