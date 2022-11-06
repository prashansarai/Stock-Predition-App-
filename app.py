import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = "2010-01-01"
end = "2021-12-31"

st.title("Stock Trend Prediction")
user_input = st.text_input('Enter Stock Ticker', 'Goog')
df = data.DataReader(user_input,"yahoo",start,end)

#Describing data
st.subheader('Data from 2010-2021')
st.write(df.describe())

#Visualisation

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Close'])
st.pyplot(fig)

# splitting data into training and testing

training_data = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
testing_data = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

#scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
training_arr = scaler.fit_transform(training_data)


#load my model
model = load_model('keras-model.h5')

#testing data

past_100_days = training_data.tail(100)
final_df = past_100_days.append(testing_data, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0],1):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i])

x_test, y_test = np.array(x_test), np.array(y_test)


# Make predictions

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

#final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'b',label ='Original price' )
plt.plot(y_predicted , 'g',label ='Predicted price' )
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Close Price', fontsize = 12)
plt.legend()
st.pyplot(fig2)






