import numpy as np
import pandas as pd
import datetime
import json
import yfinance as yf
import streamlit as st
import plotly.express as px
import mplfinance as mpf
import mplcursors
import requests as re
from datetime import date
import plotly.graph_objects as go
from yahooquery import Ticker
from datetime import datetime

from bs4 import BeautifulSoup

from pandas_datareader import data as pdr
from keras.models import load_model



st.title('Swingstock')
ticker = st.sidebar.text_input('Stock')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')




data = yf.download(ticker, start=start_date, end=end_date)
fig = px.line(data, x=data.index, y=data['Adj Close'], title=ticker)
st.plotly_chart(fig)



pricing_data, fundamental_data, technical_data, news, indication = st.tabs(["Pricing Data", "Fundamental_Data", "Technical_Data","Top News", "Indication(buy/sell/hold)"])

with pricing_data:
    st.header('Price Movements')
    data1 = data
    data1['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data1.dropna(inplace=True)
    st.write(data)

    annual_return = data1['% Change'].mean() * 252 * 100
    st.write('Annual Return = ', annual_return)

    stdev = data1["% Change"].std() * np.sqrt(252)
    st.write('Standard Deviation = ', stdev * 100, '%')
    st.write('Risk Adj. Return = ', annual_return / (stdev * 100))



    if ticker:
        sdata = Ticker(ticker)
        stock_info = sdata.summary_detail

        # Retrieving specific information
        market_cap = sdata.summary_detail[ticker]["marketCap"]/1e10
        eps = sdata.key_stats[ticker]["trailingEps"]
        pe_ratio = sdata.summary_detail[ticker]["trailingPE"]
        average_volume = sdata.summary_detail[ticker]["averageVolume10days"]
        total_shares = sdata.key_stats[ticker]["sharesOutstanding"]
        turnover = (average_volume / total_shares) * 100
        total_shares = sdata.key_stats[ticker]["sharesOutstanding"]/1e9
        dividend_yield = sdata.summary_detail[ticker]["dividendYield"]*100
        volume = sdata.summary_detail[ticker]["volume"]/1e6


        st.write(f"Market Cap: {market_cap:.2f}B")
        st.write(f"EPS: {eps:.2f}")
        st.write(f"P/E Ratio: {pe_ratio:.2f}")
        st.write(f"Turnover: {turnover:.2f}%")
        st.write(f"Total Shares: {total_shares:.7f}B")
        st.write(f"Dividend Yield: {dividend_yield:.2f}%")
        st.write(f"Volume: {volume:}M")

        for key, value in stock_info.items():
            print(f"{key}: {value}")



with fundamental_data:
    st.subheader('Financial Statements')
    statement_freq = st.radio("Select Frequency:", ("Annual", "Quarterly"))

    with st.expander("Balance Sheet"):
        if statement_freq =="Annual":
            balance_sheet = yf.Ticker(ticker).balance_sheet
        else:
            balance_sheet = yf.Ticker(ticker).quarterly_balance_sheet
        st.write(balance_sheet)

    with st.expander("Income Statement"):
        if statement_freq == "Annual":
            income_statement = yf.Ticker(ticker).income_stmt
        else:
            income_statement = yf.Ticker(ticker).quarterly_income_stmt
        st.write(income_statement)

    with st.expander("Cash Flow Statement"):
        if statement_freq == "Annual":
            cash_flow_statement = yf.Ticker(ticker).cashflow
        else:
            cash_flow_statement = yf.Ticker(ticker).quarterly_cashflow
        st.write(cash_flow_statement)


data = yf.download(ticker, start=start_date, end=end_date)
with technical_data:
    st.header('Technical Tools')
    data['Bearish Engulfing'] = np.where(
        (data['Open'].shift(1) > data['Close'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'] > data['Close'].shift(1)),
        1, 0
    )
    data['Bullish Engulfing'] = np.where(
        (data['Open'].shift(1) < data['Close'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Open'] < data['Close'].shift(1)),
        1, 0
    )


    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_gain = up.rolling(window).mean()
        avg_loss = abs(down.rolling(window).mean())
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    data['RSI'] = calculate_rsi(data)


    def calculate_rvi(data, window=10):
        data['HL'] = (data['High'] - data['Low']) / data['Close']
        data['HC'] = (data['High'] - data['Close'].shift(1)) / data['Close']
        data['LC'] = (data['Low'] - data['Close'].shift(1)) / data['Close']
        data['RVI'] = data['HL'].rolling(window).mean()
        data['RVI'] += 2 * data['HC'].rolling(window).mean()
        data['RVI'] += 2 * data['LC'].rolling(window).mean()
        data['RVI'] /= 6
        return data['RVI']


    data['RVI'] = calculate_rvi(data)

    data['Doji'] = np.where(
        (abs(data['Open'] - data['Close']) / (data['High'] - data['Low'])) < 0.1,
        1, 0
    )
    data['Morning Star'] = np.where(
        (data['Close'].shift(2) > data['Open'].shift(2)) &
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Close'].shift(1) < data['Close'].shift(2)) &
        (data['Open'] < data['Close'].shift(2)),
        1, 0
    )
    data['Evening Star'] = np.where(
        (data['Close'].shift(2) < data['Open'].shift(2)) &
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Close'].shift(1) > data['Close'].shift(2)) &
        (data['Open'] > data['Close'].shift(2)),
        1, 0
    )

    def calculate_bollinger_bands(data, window=20, std_dev=2):
        data['SMA'] = data['Close'].rolling(window).mean()
        data['Upper Band'] = data['SMA'] + (data['Close'].rolling(window).std() * std_dev)
        data['Lower Band'] = data['SMA'] - (data['Close'].rolling(window).std() * std_dev)
        return data


    data = calculate_bollinger_bands(data)

    st.write(data[['Bearish Engulfing', 'Bullish Engulfing', 'RSI', 'RVI', 'Doji', 'Morning Star', 'Evening Star','Close', 'SMA', 'Upper Band', 'Lower Band']])






with news:
    st.subheader('News')
    news_data = yf.Ticker(ticker).news

    for news_item in news_data:
        title = news_item["title"]
        publisher = news_item["publisher"]
        link = news_item["link"]
        publish_time = news_item["providerPublishTime"]
        image_url = news_item.get("image", None)

        st.title(title)
        publish_datetime = datetime.fromtimestamp(publish_time)
        publish_time = publish_datetime.strftime("%d %B %Y & %I:%M %p")

        if image_url:
            st.image(image_url)

        st.write("Publisher:", publisher)
        st.write("Publish Date & Time:", publish_time)
        st.write("Link:", link)



with indication:
    st.subheader('Indication')

    # user-selected timeframe
    timeframe = st.selectbox("Select Timeframe:", ("Hourly", "Daily", "Weekly", "Monthly"))

    # selected timeframe
    if timeframe == "Hourly":
        timeframe_data = data.resample('H').last()
    elif timeframe == "Daily":
        timeframe_data = data
    elif timeframe == "Weekly":
        timeframe_data = data.resample('W').last()
    elif timeframe == "Monthly":
        timeframe_data = data.resample('M').last()

    # latest closing price
    latest_close = timeframe_data['Close'].iloc[-1]

    user_situation = st.radio("User's Situation:", ("Already Bought", "Planning to Buy"))
    if user_situation == "Already Bought":

        # moving average20
        moving_average = timeframe_data['Close'].rolling(window=20).mean().iloc[-1]

        # RSI
        delta = timeframe_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        average_gain = gain.rolling(window=14).mean()
        average_loss = loss.rolling(window=14).mean()
        relative_strength = average_gain / average_loss
        rsi = 100 - (100 / (1 + relative_strength)).iloc[-1]

        #  Sell/hold indication
        if latest_close < moving_average and rsi > 70:
            st.write("Indication: Sell")
        else:
            st.write("Indication: Hold")
    else:

        # moving average50
        moving_average = timeframe_data['Close'].rolling(window=50).mean().iloc[-1]

        # Calculating P/E ratio
        earnings_per_share = 4.5  # Example value, replace with actual earnings per share
        price_per_share = latest_close
        pe_ratio = price_per_share / earnings_per_share

        #  Buy/wait indication
        if latest_close > moving_average and pe_ratio < 15:
            st.write("Indication: Buy")
        else:
            st.write("Indication: Wait")









