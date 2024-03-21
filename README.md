# SEM2_CA1_
##SparkContext
sc

sc.master
'local[*]'


import math
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import time
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


df = pd.read_csv('S&P_training_data.csv')

df.head()

df.shape

print(df.isnull().sum())


df.describe(include=object)
df.describe()

df.dtypes


df['Date'] = pd.to_datetime(df['Date'])

df_new = df.sort_values(by=['Date'])

df_new.head()


df_new['Price'] = pd.to_numeric(df_new['Price'])



## using a boxplot to gain insight into the data 

sns.boxplot(x=df_new['Price']) 

plt.title("Price of the S&P 500 Index 1999-2023", fontsize=14)
plt.xlabel("Price ($)", fontsize=12)

plt.figure(figsize=(16,8))
plt.title('Performance of S&P 500 Index 2018-2023')
plt.plot(df_new['Date'], df_new['Price'])
plt.xlabel('Year', fontsize=18)
plt.ylabel('Price USD ($)', fontsize=18)
plt.show


