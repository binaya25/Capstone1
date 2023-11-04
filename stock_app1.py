import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime
import yfinance as yf
import streamlit as st

from pandas_datareader import data as pdr
from keras.models import load_model


st.subheader('Date Range')

# date range
start_date_input = st.date_input('Enter Start Date')
end_date_input = st.date_input('Enter End Date')

# Converting user input to string format
start_date = start_date_input.strftime("%Y-%m-%d")
end_date = end_date_input.strftime("%Y-%m-%d")

# Timeframe selection
time_frame = st.selectbox('Select Time Frame', ('15min','1hr','4hr', '1D', '1M'))
st.title('Stock Analysis and Prediction')

user_input = st.text_input('Enter Stock Name', 'MSFT')
yf.pdr_override()
try:
    df = pdr.get_data_yahoo(user_input, start=start_date, end=end_date)
    if df.empty:
        st.write('Wrong Date Input')
    else:
        st.subheader('Data')
        st.write(df.describe())
        st.subheader('Closing Price vs Time chart')#Data visualization
        fig =plt.figure(figsize=(12,6))
        plt.plot(df.Close)
        st.pyplot(fig)


        st.subheader('Closing Price vs Time chart with 10MA')
        ma10 = df.Close.rolling(10).mean()
        fig =plt.figure(figsize=(12,6))
        plt.plot(ma10, 'g', label = 'ma10')
        plt.plot(df.Close)
        plt.legend()
        st.pyplot(fig)

        st.subheader('Closing Price vs Time chart with 10MA and 20MA')
        ma10 = df.Close.rolling(10).mean()
        ma20 = df.Close.rolling(20).mean()
        fig1 =plt.figure(figsize=(12,6))
        plt.plot(ma10, 'g', label = 'ma10')
        plt.plot(ma20, 'r', label = 'ma20')
        plt.plot(df.Close)
        plt.legend()
        st.pyplot(fig1)

        st.subheader('Closing Price vs Time chart with 10MA, 20MA and 200MA')
        ma10 = df.Close.rolling(10).mean()
        ma20 = df.Close.rolling(20).mean()
        ma200 = df.Close.rolling(200).mean()
        fig2 = plt.figure(figsize =(12,6))
        plt.plot(ma10, 'g', label= 'ma10')
        plt.plot(ma20, 'r', label= 'ma20')
        plt.plot(ma200,'#000000', label= 'ma200')
        plt.plot(df.Close)
        plt.legend(['ma10','ma20','ma200'], loc ='upper left')
        st.pyplot(fig2)

        # Train and Test split

        Train_df= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        Test_df= pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])



        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))

        Train_df_array = scaler.fit_transform(Train_df)


        #Load model
        model = load_model('model_keras_stockapp1.h5')

        #Testing
        previous_10_days = Train_df.tail(10)
        final_df = pd.concat([previous_10_days, Test_df], ignore_index=True)
        Data = scaler.fit_transform(final_df)

        X_test =[]
        Y_test =[]
        for i in range(10, Data.shape[0]):
            X_test.append(Data[i-10:i])
            Y_test.append(Data[i,0])

        X_test, Y_test = np.array(X_test), np.array(Y_test)
        y_pred = model.predict(X_test)
        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_pred = y_pred * scale_factor
        Y_test = Y_test * scale_factor



        #Final  Graph
        st.subheader('Predictions vs Original')
        fig3= plt.figure(figsize=(12,6))
        plt.plot(y_pred, 'r', label = 'Predicted Price')
        plt.plot(Y_test, 'b', label = 'Original Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig3)




        # Signal
        def crossover(ma10, ma20, ma200):
            if ma10 > ma20 and ma20 > ma200:
                return 'buy'
            elif ma10 > ma20 or ma10 > ma200:
                return 'hold'
            else:
                return 'sell'


        ma10_recent = ma10.iloc[-1]
        ma20_recent = ma20.iloc[-1]
        ma200_recent = ma200.iloc[-1]
        signal = crossover(ma10_recent, ma20_recent, ma200_recent)

        st.subheader('Trading Signal')
        st.write('Indication :', signal)

        # Button
        if st.button('Why?'):
            if signal == 'buy':
                st.write(
                    "A 'buy' signal indicates that the stock's moving averages suggest a positive trend, and it may be a good time to consider purchasing the stock.")
                st.write("The moving averages used in this analysis are calculated as follows:")
                st.write(
                    "- ma10: The 10-day moving average is the average closing price of the stock over the past 10 trading days.")
                st.write(
                    "- ma20: The 20-day moving average is the average closing price of the stock over the past 20 trading days.")
                st.write(
                    "- ma200: The 200-day moving average is the average closing price of the stock over the past 200 trading days.")
                st.write(
                    "When the ma10 crosses above both ma20 and ma200, it suggests a positive trend as recent prices are higher than the average prices over the short, medium, and long term.")
            elif signal == 'hold':
                st.write(
                    "A 'hold' signal indicates that the stock's moving averages suggest a relatively stable trend, and it may be prudent to continue holding the stock without making any immediate buying or selling decisions.")
                st.write("The moving averages used in this analysis are calculated as follows:")
                st.write(
                    "- ma10: The 10-day moving average is the average closing price of the stock over the past 10 trading days.")
                st.write(
                    "- ma20: The 20-day moving average is the average closing price of the stock over the past 20 trading days.")
                st.write(
                    "- ma200: The 200-day moving average is the average closing price of the stock over the past 200 trading days.")
                st.write(
                    "When the ma10 is above ma200 or ma20, it suggests a relatively stable trend, indicating that recent prices are in line with the average prices over the short or medium term.")
            else:
                st.write(
                    "A 'sell' signal indicates that the stock's moving averages suggest a negative trend, and it may be a good time to consider selling the stock.")
                st.write("The moving averages used in this analysis are calculated as follows:")
                st.write(
                    "- ma10: The 10-day moving average is the average closing price of the stock over the past 10 trading days.")
                st.write(
                    "- ma20: The 20-day moving average is the average closing price of the stock over the past 20 trading days.")
                st.write(
                    "- ma200: The 200-day moving average is the average closing price of the stock over the past 200 trading days.")
                st.write(
                    "When the ma10 crosses below both ma20 and ma200, it suggests a negative trend as recent prices are lower than the average prices over the short, medium, and long term.")
            st.write(
                "The closing price (df.Close) is the last traded price of the stock at the end of each trading day. It is used to analyze the historical price movements of the stock and calculate the moving averages.")
except ValueError:
    st.write('Wrong Date Input')

