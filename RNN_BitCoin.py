# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading CSV file from train set
training_set = pd.read_csv('Enter the name of file for training_set ')
training_set.head()

#Selecting the second column [for prediction]
training_set = training_set.iloc[:,1:2]
training_set.head()

# Converting into 2D array
training_set = training_set.values
training_set

# Scaling of Data [Normalization]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
training_set = sc.fit_transform(training_set)
training_set

# Lenth of the Data set
len(training_set)
898

X_train = training_set[0:897]
Y_train = training_set[1:898]

# X_train must be equal to Y_train
len(X_train)
897
len(Y_train)
897

#Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]
X_train = np.reshape(X_train, (897, 1, 1))
X_train

#-------------------------Need to be have Keras and TensorFlow backend--------------------------- 


#RNN Layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the Recurrent Neural Network
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the Recurrent Neural Network [epoches is a kindoff number of iteration]
regressor.fit(X_train, Y_train, batch_size = 32, epochs = 200)


# Reading CSV file from test set
test_set = pd.read_csv('Enter the name of file for testing_set')
test_set.head()

#selecting the second column from test data 
real_btc_price = test_set.iloc[:,1:2]         

# Coverting into 2D array
real_btc_price = real_btc_price.values      

#getting the predicted BTC value of the first week of Dec 2017  
inputs = real_btc_price			
inputs = sc.transform(inputs)

#Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]
inputs = np.reshape(inputs, (8, 1, 1))
predicted_btc_price = regressor.predict(inputs)
predicted_btc_price = sc.inverse_transform(predicted_btc_price)

#Graphs for predicted values
plt.plot(real_btc_price, color = 'red', label = 'Real BTC Value')
plt.plot(predicted_btc_price, color = 'blue', label = 'Predicted BTC Value')
plt.title('BTC Value Prediction')
plt.xlabel('Days')
plt.ylabel('BTC Value')
plt.legend()
plt.show()

