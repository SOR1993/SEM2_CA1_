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

data = df_new.filter(['Price'])

dataset = data.values

training_data_len = math.ceil( len(dataset) * .8 )

training_data_len

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

#scaled_data
dataset

train_data = dataset[0:training_data_len , :]

x_train =[]
y_train =[]

for i in range(20, len(train_data)):
    x_train.append(train_data[i-20:i,0])
    y_train.append(train_data[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train =np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))
x_train.shape

model = Sequential()
model.add(LSTM(10, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(10, return_sequences = False))
model.add(Dense(5))
model.add(Dense(1))

#model.compile(optimizer='adam', loss = 'mean_squared_error')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = 5, epochs = 10)

test_data = dataset[training_data_len -20: , :]

x_test = []
y_test = dataset[training_data_len:, :]

for i in range(20,len(test_data)):
    x_test.append(test_data[i-20:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(16,8))
plt.title('Predictions of Performance of the S&P 500 Index', fontsize = 18)
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Price ($)', fontsize = 14)
plt.plot(predictions)

rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test),predictions))
r2_value = r2_score(scaler.inverse_transform(y_test),predictions)

print("Root Mean Square Error: ", rmse)
print("R^2 Value: ", r2_value)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

train.head()
valid.head()

df_concat = pd.concat([train, valid], axis=0)

plt.figure(figsize=(16,8))
plt.title('LSTM Deep Learning Neural Networks Model', fontsize=20)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Price USD ($)', fontsize=18)
plt.plot(train['Price'])
plt.plot(valid[['Price', 'Predictions']])
plt.legend(['Train', 'Valid', 'Predictions'], loc = 'lower right')
plt.show()

stock_price_pred = pd.read_csv('S&P_training_data.csv')

new_df = stock_price_pred['Price']

last_20_days = new_df.head(20).values

last_20_days_scaled = scaler.transform(last_20_days.reshape(-1, 1))

X_test2 = []

X_test2.append(last_20_days_scaled)

X_test2 = np.array(X_test2)

X_test2 = np.reshape(X_test2,(X_test2.shape[0],X_test2.shape[1], 1))

pred_price = model.predict(X_test2)

pred_price = scaler.inverse_transform(pred_price)
print('The predicted price of the S&P500 Index 20 Days from the end, is: ', pred_price)  

stock_price_pred.head(20)





## Model 2 (TimeStep=40)
# we create a df 'data' with just the price data 
data = df.filter(['Price'])
# the dataframe is then converted to a numpy array (A2 dimensional matrix)
dataset = data.values
# looking to use an 80-20 training-validation split of the data
training_data_len = math.ceil( len(dataset) * .8 )
# looking at how many rows of data will be used to train the model (80%=1208 rows)
training_data_len
# compute the min and max values to be used for scaling
# transforms the data based on the two values computed
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
# this is what the scaled data looks like
# ranges between 0 and 1 inclusive
dataset
# create the trained data set (Length of the training set is 80% of the total dataset as outlined earlier)
train_data = dataset[0:training_data_len , :]
# create two arrays to hold the training data
x_train =[] # independent variables
y_train =[] # dependent variable
# create a loop to iterate through the training data to split the data into X_train and y_train
# the data has been split into iterations of 20 in line with there being approx. 20 trading days per month
# this is a hyperparamter which can be tuned depending on the aim of the research
for i in range(40, len(train_data)):
    x_train.append(train_data[i-40:i,0])
    y_train.append(train_data[i,0])
# convert x_train and y_train to numpy arrays to train the LTSM Neural Net Model
x_train, y_train = np.array(x_train), np.array(y_train)
# LTSM Neural Model expects three dimensions; number of samples (1188), number of time steps (20) and number of features columns (1)
x_train =np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))
x_train.shape
# build the LTSM Neural Net Model
model = Sequential()
model.add(LSTM(10, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(10, return_sequences = False))
model.add(Dense(5))
model.add(Dense(1))
#model.compile(optimizer='adam', loss = 'mean_squared_error')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# train the model using the x_train and y_train data
model.fit(x_train, y_train, batch_size = 5, epochs = 10)
# create the test data set
test_data = dataset[training_data_len -40: , :]
# create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
# now we loop to iterate through the data contained in x_test and y_test
for i in range(40,len(test_data)):
    x_test.append(test_data[i-40:i, 0])
# convert the data to a numpy array
x_test = np.array(x_test)
# reshape the data as the LTSM Model will expect a three dimensional shape as outliend previously
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))
############
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# calculate the RMSE and R2 (Goodness of Fit)
rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test),predictions))
r2_value = r2_score(scaler.inverse_transform(y_test),predictions)
print("Root Mean Square Error: ", rmse)
print("R^2 Value: ", r2_value)
# create a new dataframe to show the actual and predicted prices
stock_price_pred = pd.read_csv('S&P_training_data.csv')
new_df = stock_price_pred['Price']
# get the last 20 days of data
last_20_days = new_df.head(20).values
# convert the dataframe to an array
last_20_days_scaled = scaler.transform(last_20_days.reshape(-1, 1))
# create an empty list
X_test2 = []
# append the last 20 days to X_test2
X_test2.append(last_20_days_scaled)
# convert to a numpy array
X_test2 = np.array(X_test2)
#reshape the data
X_test2 = np.reshape(X_test2,(X_test2.shape[0],X_test2.shape[1], 1))
# get the models prediction
pred_price = model.predict(X_test2)
# undo scaling and print the prediction
pred_price = scaler.inverse_transform(pred_price)
print('The predicted price of the S&P500 Index 20 Days from the end, is: ', pred_price)  
#compare vs the actual values
stock_price_pred.head(20)





