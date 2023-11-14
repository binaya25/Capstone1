import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime
import yfinance as yf
import streamlit as st

from pandas_datareader import data as pdr
from keras.models import load_model




# Input date range
start_date_input = st.date_input('Enter Start Date')
end_date_input = st.date_input('Enter End Date')

# Convert the user input to string format
start_date = start_date_input.strftime("%Y-%m-%d")
end_date = end_date_input.strftime("%Y-%m-%d")


st.title('Stock Analysis and Prediction')

user_input = st.text_input('Enter Stock Name', 'MSFT')
yf.pdr_override()
df = pdr.get_data_yahoo(user_input, start=start_date, end=end_date)


#Data
st.subheader('Date Range')
st.write(df.describe())

#Data visualization
st.subheader('Closing Price vs Time chart')
fig =plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig =plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig =plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Train and Test split

Train_df= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
Test_df= pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

Train_df_array = scaler.fit_transform(Train_df)


#Load model
model = load_model('model_keras.h5')

#Testing
previous_100_days = Train_df.tail(100)
final_df = pd.concat([previous_100_days, Test_df], ignore_index=True)
Data = scaler.fit_transform(final_df)

X_test =[]
Y_test =[]

for i in range(100, Data.shape[0]):
    X_test.append(Data[i-100:i])
    Y_test.append(Data[i,0])

X_test, Y_test = np.array(X_test), np.array(Y_test)
y_pred = model.predict(X_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_pred = y_pred * scale_factor
Y_test = Y_test * scale_factor



#Final  Graph
st.subheader('Predictions vs Original')
fig1= plt.figure(figsize=(12,6))
plt.plot(y_pred, 'r', label = 'Predicted Price')
plt.plot(Y_test, 'b', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)




