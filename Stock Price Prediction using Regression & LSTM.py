#!/usr/bin/env python
# coding: utf-8

# # Understanding the Problem Statement & Business Case
# 
# - Artificial Intelligence (AI), Machine Learning (ML) and Deep Learning (DL) have been transforming finance and investing.
# - "Artificial intelligence is to trading what fire was to the cavemen"!
# - Electronic trades account for almost 45% of revenues in cash equities trading"
# - AI powered robo-advisers can perform real-time analysis on massive datasets and trade securities at an extremely faster rate compared to human traders.
# - AI-powered trading could potentially reduce risk and maximize returns.
# - In this project, I trained a ridge regression model and deep neural network model to predict future stock prices.
# - By accurately predicting stock prices, investors can maximize returns and know when to buy/sell securities.
# - The AI/ML model will be trained using historical stock price data along with the volume of transactions.
# - We will use a type of neural nets known as Long Short-Term Memory Networks (LSTM).
# - Disclaimer: Stock prices are volatile and are generally hard to predict. Invest at your own risk.

# # Import Datasets and Libraries

# In[1]:


import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


# In[2]:


# Read stock prices data
stock_price_df = pd.read_csv("D:\Python and Machine Learning for Financial Analysis\stock.csv")
stock_price_df


# In[3]:


# Read the stocks volume data
stock_vol_df = pd.read_csv("D:\Python and Machine Learning for Financial Analysis\stock_volume.csv")
stock_vol_df


# In[4]:


# Sorted the data based on Date
stock_price_df = stock_price_df.sort_values(by = ['Date'])
stock_price_df


# In[5]:


# Sorted the data based on Date
stock_vol_df = stock_vol_df.sort_values(by = ['Date'])
stock_vol_df


# In[6]:


# Checked if Null values exist in stock prices data
stock_price_df.isnull().sum()


# In[7]:


# Checked if Null values exist in stocks volume data
stock_vol_df.isnull().sum()


# In[8]:


# Got stock prices dataframe info
stock_price_df.info()


# In[9]:


# Got stock volume dataframe info
stock_vol_df.info()


# In[10]:


# Got the statistical data for the stocks volume dataframe
stock_vol_df.describe()


# - Average trading volume for Apple stock is 5.820332e+07
# 
# - Maximum trading volume for S&P500 is 9.044690e+09
# 
# - S&P500 is the most traded security of them all.
# - The S&P 500 index is a broad-based measure of large corporations traded on U.S. stock markets. Over long periods of time, passively holding the index often produces better results than actively trading or picking single stocks.
# - Over long-time horizons, the index typically produces better returns than actively managed portfolios.

# In[11]:


# Got the statistical data for the prices dataframe
stock_price_df.describe()


# - Average Stock Price for S&P500 = 2218.749554
# - Maximum Tesla Stock Price = 1643.000000

# # Performed Exploratory Data Analysis and Visualization

# In[12]:


# Function to normalize stock prices based on their initial price
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x


# In[13]:


# Function to plot interactive plots using Plotly Express
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()


# In[14]:


# Plotted interactive chart for stocks data
interactive_plot(stock_price_df, 'Stock Prices')


# In[15]:


# Plotted interactive chart for volume data
# Noticed that S&P500 trading is orders of magnitude compared to individual stocks
interactive_plot(stock_vol_df, 'Stocks Volume')


# In[16]:


# Plotted interactive chart for normalized stocks prices data
interactive_plot(normalize(stock_price_df), 'Stock Prices')

# Let's normalize the data and re-plot interactive chart for volume data
interactive_plot(normalize(stock_vol_df), 'Normalized Volume')


# # Prepared the Data before Training the AI/ML Model
# 
# ## Training & Testing Data Split
# 
# + Data set is divided into 75% for training and 25% for testing.
#     - Training set: used for model training.
#     - Testing set: used for testing trained model. Made sure that the testing dataset has never been seen by the trained model before.

# In[17]:


# Function to concatenate the date, stock price, and volume in one dataframe
def individual_stock(price_df, vol_df, name):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name], 'Volume': vol_df[name]})


# In[18]:


# Function to return the input/output (target) data for AI/ML Model
# Noted that our goal is to predict the future stock price 
# Target stock price today will be tomorrow's price 
def trading_window(data):
  
  # 1 day window 
  n = 1

  # Created a column containing the prices for the next 1 days
  data['Target'] = data[['Close']].shift(-n)
  
  # Returned the new dataset 
  return data


# In[19]:


# Let's test the functions and get individual stock prices and volumes for AAPL
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')
price_volume_df


# In[20]:


price_volume_target_df = trading_window(price_volume_df)
price_volume_target_df


# In[21]:


# Removed the last row as it will be a null value
price_volume_target_df = price_volume_target_df[:-1]
price_volume_target_df


# In[22]:


# Scaled the data
sc = MinMaxScaler(feature_range = (0, 1))
price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns = ['Date']))


# In[23]:


price_volume_target_scaled_df


# In[24]:


price_volume_target_scaled_df.shape


# In[25]:


# Created Feature and Target
X = price_volume_target_scaled_df[:,:2]
y = price_volume_target_scaled_df[:,2:]


# In[26]:


# Converted dataframe to arrays
# X = np.asarray(X)
# y = np.asarray(y)
X.shape, y.shape


# In[27]:


# Split the data this way, since order is important in time-series
# Noted that we did not use train test split with it's default settings since it shuffles the data
split = int(0.65 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]


# In[28]:


X_train.shape, y_train.shape


# In[29]:


X_test.shape, y_test.shape


# In[30]:


# Defined a data plotting function
def show_plot(data, title):
  plt.figure(figsize = (13, 5))
  plt.plot(data, linewidth = 3)
  plt.title(title)
  plt.grid()

show_plot(X_train, 'Training Data')
show_plot(X_test, 'Testing Data')


# In[31]:


# Let's test the functions and get individual stock prices and volumes for S&P500
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'sp500')
price_volume_df


# In[32]:


# Let's test the functions and get individual stock prices and volumes for Amazon 
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AMZN')
price_volume_df


# # Understanding the Theory and Intuition behind Regression
# 
# ## Simple Linear Regression: Intuition
# 
# - In simple linear regression, we predict the value of one variable Y based on another variable X.
# - X is called the independent variable and Y is called the dependant variable.
# - It is called simple because it examines relationship between two variables only.
# - It is called linear as when the independent variable increases (or decreases), the dependent variable increases (or decreases) in a linear fashion.
# 
# ## M and B
# 
# - Once the coefficients m and b are obtained, you have obtained a simple linear regression model!
# - This "trained" model can be later used to predict.
# 
# ## Simple Linear Regression: To Obtain Model Parameters using Least Sum of Squares
# 
# - Least squares fitting is a way to find the best fit curve or line for a set of points.
# - The sum of the squares of the offsets (residuals) are used to estimate the best fit curve or line.
# - Least squares method is used to obtain the coefficients m and b.
# 
# # Understanding the Concept of Regularization & Ridge Regression
# 
# ## Regularization: Intuition
# 
# - Regularization techniques are used to avoid networks overfitting
# - Overfitting occurs when the model provide great results on the training data but performs poorly on testing dataset.
# - Overfitting occurs when the model learns all the patterns of the training dataset but fails to generalize.
# - Overfitted models generally provide high accuracy on training dataset but low accuracy on testing and validation (evaluation) datasets
# 
# ## Ridge Regression (L2 Regularization): Intuition
# 
# - Ridge regression advantage is to avoid overfitting.
# - Our ultimate model is the one that could generalize patterns; i.e.: works best on the training and testing dataset
# - Overfitting occurs when the trained model performs well on the training data and performs poorly on the testing datasets
# - Ridge regression works by applying a penalizing term (reducing the weights and biases) to overcome overfitting.
# - Ridge regression works by attempting at increasing the bias to improve variance (generalization capability)
# - This works by changing the slope of the line
# - The model performance might become little poor on the training set but it will perform consistently well on both the training and testing datasets.
# 
# ## Ridge Regression (L2 Regularization): Math
# 
# - Slope when reduced with the ridge regression penalty makes the model become less sensitive to changes in the independent variable
# 
# ## Ridge Regression (L2 Regularization): Alpha Effect
# 
# - As Alpha increases, the slope of the regression line reduces and becomes more horizontal.
# - As Alpha increases, the model becomes less sensitive to the variations of the independent variable
# 
# # Built and Trained a Ridge Linear Regression Model

# In[33]:


# Noted that Ridge regression performs linear least squares with L2 regularization.
# Created and trained the Ridge Linear Regression  Model
regression_model = Ridge()
regression_model.fit(X_train, y_train)


# In[34]:


# Tested the model and calculated its accuracy 
lr_accuracy = regression_model.score(X_test, y_test)
print("Linear Regression Score: ", lr_accuracy)


# In[35]:


# Made Prediction
predicted_prices = regression_model.predict(X)
predicted_prices


# In[36]:


# Append the predicted values into a list
Predicted = []
for i in predicted_prices:
  Predicted.append(i[0])


# In[37]:


len(Predicted)


# In[38]:


# Append the close values to the list
close = []
for i in price_volume_target_scaled_df:
  close.append(i[0])


# In[39]:


# Created a dataframe based on the dates in the individual stock data
df_predicted = price_volume_target_df[['Date']]
df_predicted


# In[40]:


# Added the close values to the dataframe
df_predicted['Close'] = close
df_predicted


# In[41]:


# Added the predicted values to the dataframe
df_predicted['Prediction'] = Predicted
df_predicted


# In[42]:


# Plotted the results
interactive_plot(df_predicted, "Original Vs. Prediction")


# In[43]:


# Noted that Ridge regression performs linear least squares with L2 regularization.
# Created and trained the Ridge Linear Regression  Model
regression_model_2 = Ridge(alpha = 2)
regression_model_2.fit(X_train, y_train)

# Tested the model and calculated its accuracy 
lr_accuracy = regression_model_2.score(X_test, y_test)
print("Linear Regression Score: ", lr_accuracy)

# Made Prediction
predicted_prices = regression_model_2.predict(X)
predicted_prices

# Append the predicted values into a list
Predicted = []
for i in predicted_prices:
  Predicted.append(i[0])

# Append the close values to the list
close = []
for i in price_volume_target_scaled_df:
  close.append(i[0])

# Created a dataframe based on the dates in the individual stock data
df_predicted_2 = price_volume_target_df[['Date']]
df_predicted_2

# Added the close values to the dataframe
df_predicted_2['Close'] = close
df_predicted_2

# Added the predicted values to the dataframe
df_predicted_2['Prediction'] = Predicted
df_predicted_2

# Plotted the results
interactive_plot(df_predicted_2, "Original Vs. Prediction")


# # Understanding the Theory and Intuition behind Neural Networks
# 
# ## Neuron Mathematical Model
# 
# - Artificial Neural Networks are information processing models that are inspired by the human brain.
# - The neuron collects signals from input channels named dendrites, processes information in its nucleus, and then generates an output in a long thin branch called axon.
# 
# # Understanding how do Artificial Neural Networks Train
# 
# ## ANN Training using Gradient Descent
# 
# - Gradient descent is an optimization algorithm used to obtain the optimized network weight and bias values.
# - It works by iteratively trying to minimize the cost function.
# - It works by calculating the gradient of the cost function and moving in the negative direction until the local/global minimum is achieved.
# - The size of the steps taken are called the learning rate
# - If learning rate increases, the area covered in the search space will increase so we might reach global minimum faster
# - However, we can overshoot the target
# - For small learning rates, training will take much longer to reach optimized weights values
# 
# # Understanding the Theory and Intuition behind Recurrent Neural Networks
# 
# ## Recurrent Neural Networks (RNN):
# 
# - Feedforward Neural Networks (vanilla networks) map a fixed size input (such as image) to a fixed size output (classes or probabilities).
# - A drawback in Feedforward networks is that they do not have any time dependency or memory effect.
# - A RNN is a type of ANN that is designed to take temporal dimension into consideration by having a memory (internal state) (feedback loop).
# 
# ## RNN Architecture
# 
# - A RNN contains a temporal loop in which the hidden layer not only gives an output but it feeds itself as well.
# - An extra dimension is added which is time!
# - RNN can recall what happened in the previous time stamp so it works great with sequence of text.
# 
# ## RNNs are Special
# 
# - Feedforward ANNs are so constrained with their fixed number of input and outputs.
# - For example, a CNN will have fixed size image and generates a fixed output (class or probabilities).
# - Feedforward ANN have a fixed configuration, i.e.: same number of hidden layers and weights.
# - Recurrent Neural Networks offer huge advantage over feedforward ANN and they are much more fun!
# + RNN allows us to work with a sequence of vectors:
#     - Sequence in inputs
#     - Sequence in outputs
#     - Sequence in both!
# 
# ## RNN Math
# 
# - A RNN accepts an input x and generate an output o.
# - The output o does not depend on the input x alone, however, it depends on the entire history of the inputs that have been fed to the network in previous time steps.
# 
# # Understanding the Theory and Intuition behind Long Short Term Memory Networks
# 
# ## Vanishing Gradient Problem
# 
# - LSTM networks work much better compared to vanilla RNN since they overcome the vanishing gradient problem.
# - The error has to propogate through all the previous layers resulting in a vanishing gradient.
# - As the gradient goes smaller, the network weights are no longer updated.
# - As more layers are added, the gradients of the loss function approaches zero, making the network hard to train.
# - ANN gradients are calculated during backpropagation.
# - In backpropagation, we calculate the derivatives of the network by moving from the outermost layer (close to output) back to the initial layers (close to inputs).
# - The chain rule is used during this calculation in which the derivatives from the final layers are multiplied by the derivaties from early layers.
# - The gradients keeps diminishing exponentially and therefore the weights and biases are no longer being updated.
# 
# ## LSTM Intuition
# 
# - LSTM networks work better compared to vanilla RNN since they overcome vanishing gradient problem.
# - In practice, RNN fail to establish long term dependencies.
# - LSTM networks are type of RNN that are designed to remember long term dependencies by default.
# - LSTM can remember and recall information for a prolonged period of time.
# 
# ## LSTM Intuition - Gates
# 
# - LSTM contains gates that can allow or block information passing by.
# - Gates consist of a sigmoid neural net layer along with a pointwise multiplication operation.
# + Sigmoid output ranges from 0 to 1:
#     - 0 = Don't allow any data to flow
#     - 1 = Allow everything to flow!
# 
# # Trained an LSTM Time Series Model

# In[44]:


# Let's test the functions and get individual stock prices and volumes for AAPL
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')
price_volume_df


# In[45]:


# Got the close and volume data as training data (Input)
training_data = price_volume_df.iloc[:, 1:3].values
training_data


# In[46]:


# Normalized the data
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_data)


# In[47]:


# Created the training and testing data, training data contains present day and previous day values
X = []
y = []
for i in range(1, len(price_volume_df)):
    X.append(training_set_scaled [i-1:i, 0])
    y.append(training_set_scaled [i, 0])


# In[48]:


X


# In[49]:


# Converted the data into array format
X = np.asarray(X)
y = np.asarray(y)


# In[50]:


# Splitted the data
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]


# In[51]:


# Reshaped the 1D arrays to 3D arrays to feed in the model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape


# In[52]:


# Created the model
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()


# In[53]:


# Trained the model
history = model.fit(
    X_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)


# In[54]:


# Made prediction
predicted = model.predict(X)


# In[55]:


# Append the predicted values to the list
test_predicted = []

for i in predicted:
  test_predicted.append(i[0])


# In[56]:


test_predicted


# In[57]:


df_predicted = price_volume_df[1:][['Date']]
df_predicted


# In[58]:


df_predicted['predictions'] = test_predicted


# In[59]:


df_predicted


# In[60]:


# Plot the data
close = []
for i in training_set_scaled:
  close.append(i[0])


# In[61]:


df_predicted['Close'] = close[1:]


# In[62]:


df_predicted


# In[63]:


# Plot the data
interactive_plot(df_predicted, "Original Vs Prediction")


# In[64]:


# Let's test the functions and get individual stock prices and volumes for sp500
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'sp500')
price_volume_df

# Get the close and volume data as training data (Input)
training_data = price_volume_df.iloc[:, 1:3].values
training_data

# Normalize the data
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_data)

# Created the training and testing data, training data contains present day and previous day values
X = []
y = []
for i in range(1, len(price_volume_df)):
    X.append(training_set_scaled [i-1:i, 0])
    y.append(training_set_scaled [i, 0])
    
# Converted the data into array format
X = np.asarray(X)
y = np.asarray(y)

# Splitted the data
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshaped the 1D arrays to 3D arrays to feed in the model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape

# Created the model
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()

# Trained the model
history = model.fit(
    X_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)

# Made prediction
predicted = model.predict(X)

# Append the predicted values to the list
test_predicted = []

for i in predicted:
  test_predicted.append(i[0])

test_predicted

df_predicted = price_volume_df[1:][['Date']]
df_predicted

df_predicted['predictions'] = test_predicted

df_predicted

# Plot the data
close = []
for i in training_set_scaled:
  close.append(i[0])

df_predicted['Close'] = close[1:]

df_predicted

# Plot the data
interactive_plot(df_predicted, "Original Vs Prediction")


# In[65]:


# Let's test the functions and get individual stock prices and volumes for AMZN
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AMZN')
price_volume_df

# Get the close and volume data as training data (Input)
training_data = price_volume_df.iloc[:, 1:3].values
training_data

# Normalize the data
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_data)

# Created the training and testing data, training data contains present day and previous day values
X = []
y = []
for i in range(1, len(price_volume_df)):
    X.append(training_set_scaled [i-1:i, 0])
    y.append(training_set_scaled [i, 0])
    
# Converted the data into array format
X = np.asarray(X)
y = np.asarray(y)

# Splitted the data
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshaped the 1D arrays to 3D arrays to feed in the model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape

# Created the model
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()

# Trained the model
history = model.fit(
    X_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)

# Made prediction
predicted = model.predict(X)

# Append the predicted values to the list
test_predicted = []

for i in predicted:
  test_predicted.append(i[0])

test_predicted

df_predicted = price_volume_df[1:][['Date']]
df_predicted

df_predicted['predictions'] = test_predicted

df_predicted

# Plot the data
close = []
for i in training_set_scaled:
  close.append(i[0])

df_predicted['Close'] = close[1:]

df_predicted

# Plot the data
interactive_plot(df_predicted, "Original Vs Prediction")


# In[66]:


# Let's test the functions and get individual stock prices and volumes for TSLA
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'TSLA')
price_volume_df

# Get the close and volume data as training data (Input)
training_data = price_volume_df.iloc[:, 1:3].values
training_data

# Normalize the data
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_data)

# Created the training and testing data, training data contains present day and previous day values
X = []
y = []
for i in range(1, len(price_volume_df)):
    X.append(training_set_scaled [i-1:i, 0])
    y.append(training_set_scaled [i, 0])
    
# Converted the data into array format
X = np.asarray(X)
y = np.asarray(y)

# Splitted the data
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshaped the 1D arrays to 3D arrays to feed in the model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape

# Created the model
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()

# Trained the model
history = model.fit(
    X_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)

# Made prediction
predicted = model.predict(X)

# Append the predicted values to the list
test_predicted = []

for i in predicted:
  test_predicted.append(i[0])

test_predicted

df_predicted = price_volume_df[1:][['Date']]
df_predicted

df_predicted['predictions'] = test_predicted

df_predicted

# Plot the data
close = []
for i in training_set_scaled:
  close.append(i[0])

df_predicted['Close'] = close[1:]

df_predicted

# Plot the data
interactive_plot(df_predicted, "Original Vs Prediction")


# In[67]:


# Let's test the functions and get individual stock prices and volumes for AAPL
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')
price_volume_df

# Get the close and volume data as training data (Input)
training_data = price_volume_df.iloc[:, 1:3].values
training_data

# Normalize the data
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_data)

# Created the training and testing data, training data contains present day and previous day values
X = []
y = []
for i in range(1, len(price_volume_df)):
    X.append(training_set_scaled [i-1:i, 0])
    y.append(training_set_scaled [i, 0])
    
# Converted the data into array format
X = np.asarray(X)
y = np.asarray(y)

# Splitted the data
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshaped the 1D arrays to 3D arrays to feed in the model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape

# Created the model
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()

# Trained the model
history = model.fit(
    X_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)

# Made prediction
predicted = model.predict(X)

# Append the predicted values to the list
test_predicted = []

for i in predicted:
  test_predicted.append(i[0])

test_predicted

df_predicted = price_volume_df[1:][['Date']]
df_predicted

df_predicted['predictions'] = test_predicted

df_predicted

# Plot the data
close = []
for i in training_set_scaled:
  close.append(i[0])

df_predicted['Close'] = close[1:]

df_predicted

# Plot the data
interactive_plot(df_predicted, "Original Vs Prediction")

