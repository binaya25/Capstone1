import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import streamlit as st
import plotly.express as px
from keras.models import load_model
import nltk
from datetime import date
import plotly.graph_objects as go
from yahooquery import Ticker
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

st.title('Swingstock')


ticker = st.sidebar.text_input('Stock')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')


if ticker =='':
    st.sidebar.write('Input Stock Name')



# ticker = input("Please enter the stock's ticker symbol: ")
# start_date = input("Please enter the start date (YYYY-MM-DD): ")
# end_date = input("Please enter the end date (YYYY-MM-DD): ")
#
# if not ticker or not start_date or not end_date:
#     print("Please input the stock's name and date.")
# else:
#     data = yf.download(ticker, start=start_date, end=end_date)




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
        market_cap = stock_info[ticker]["marketCap"]/1e10
        eps = sdata.key_stats[ticker]["trailingEps"]
        pe_ratio = stock_info[ticker]["trailingPE"]
        average_volume = stock_info[ticker]["averageVolume10days"]
        total_shares = sdata.key_stats[ticker]["sharesOutstanding"]
        turnover = (average_volume / total_shares) * 100
        total_shares = sdata.key_stats[ticker]["sharesOutstanding"]/1e9
        # dividend_yield = stock_info[ticker]["dividendYield"] * 100
        volume = stock_info[ticker]["volume"]/1e6







        st.write(f"Market Cap: {market_cap:.2f}B")
        st.write(f"EPS: {eps:.2f}")
        st.write(f"P/E Ratio: {pe_ratio:.2f}")
        st.write(f"Turnover: {turnover:.2f}%")
        st.write(f"Total Shares: {total_shares:.7f}B")
        # st.write(f"Dividend Yield: {dividend_yield:.2f}%")
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
    def perform_sentiment_analysis(articles): # sentiment analysis using VADER
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = []
        for article in articles:
            sentiment_scores.append(sid.polarity_scores(article))
        return sentiment_scores


    # Calculate overall sentiment score
    def calculate_sentiment_score(sentiment_scores):
        compound_scores = [score['compound'] for score in sentiment_scores]
        overall_score = sum(compound_scores) / len(compound_scores)
        return overall_score


    # buy/sell indication based on sentiment score
    def provide_buy_sell_indication(score):
        if score >= 0.1:
            return "Buy"
        elif score <= -0.1:
            return "Sell"
        else:
            return "Hold"

    # Fetching news
    news_data = yf.Ticker(ticker).news
    news_articles = [news_item["title"] for news_item in news_data]

    # Sentiment analysis
    sentiment_scores = perform_sentiment_analysis(news_articles)
    overall_score = calculate_sentiment_score(sentiment_scores)
    buy_sell_indication = provide_buy_sell_indication(overall_score)

    st.subheader('News')

    for i, news_item in enumerate(news_data):
        title = news_item["title"]
        publisher = news_item["publisher"]
        link = news_item["link"]
        publish_time = news_item["providerPublishTime"]
        image_url = news_item.get("image", None)

        st.title(f"News {i + 1}: {title}")
        publish_datetime = datetime.fromtimestamp(publish_time)
        publish_time = publish_datetime.strftime("%d %B %Y & %I:%M %p")

        if image_url:
            st.image(image_url)

        st.write("Publisher:", publisher)
        st.write("Publish Date & Time:", publish_time)
        st.write("Link:", link)

    st.subheader('Sentiment Analysis')
    st.write("Overall Sentiment Score: ", overall_score)
    st.write("Buy/Sell Indication: ", buy_sell_indication)


with indication:
    st.subheader('Indication')
    timeframe = st.selectbox("Select Timeframe:", ("Hourly", "Daily", "Weekly", "Monthly"))

    # Selected timeframe
    if timeframe == "Hourly":
        timeframe_data = data.resample('H').last()
    elif timeframe == "Daily":
        timeframe_data = data
    elif timeframe == "Weekly":
        timeframe_data = data.resample('W').last()
    elif timeframe == "Monthly":
        timeframe_data = data.resample('M').last()

    # Latest closing price
    latest_close = timeframe_data['Close'].iloc[-1]

    user_situation = st.radio("User's Situation:", ("Already Bought", "Planning to Buy"))
    if user_situation == "Already Bought":

        # Moving average20
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
            st.write("Overall: Sell")
        else:
            st.write("Overall: Hold")
    else:

        # Moving average50
        moving_average = timeframe_data['Close'].rolling(window=50).mean().iloc[-1]

        # Calculating P/E ratio
        earnings_per_share = 4.5
        price_per_share = latest_close
        pe_ratio = price_per_share / earnings_per_share

        #  Buy/wait indication
        if latest_close > moving_average and pe_ratio < 15:
            st.write("Overall: Buy")
        else:
            st.write("Overall: Wait")


    ma10 = data.Close.rolling(10).mean()
    ma20 = data.Close.rolling(20).mean()
    ma200 = data.Close.rolling(200).mean()

    # Train_df = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
    # Test_df = pd.DataFrame(data['Close'][int(len(data) * 0.70):int(len(data))])



    # from sklearn.preprocessing import MinMaxScaler
    #
    # scaler = MinMaxScaler(feature_range=(0, 1))
    #
    # Train_df_array = scaler.fit_transform(Train_df)



    # Load model
    # model = load_model('model_keras_stockapp1.h5')
    #
    # # Testing
    # previous_10_days = Train_df.tail(10)
    # final_df = pd.concat([previous_10_days, Test_df], ignore_index=True)
    # Data = scaler.fit_transform(final_df)
    #
    # X_test = []
    # Y_test = []
    #
    # for i in range(10, Data.shape[0]):
    #     X_test.append(Data[i - 10:i])
    #     Y_test.append(Data[i, 0])
    #
    # X_test = np.array(X_test)
    # Y_test = np.array(Y_test)
    #
    # y_pred = model.predict(X_test)
    # scaler = scaler.scale_
    #
    # scale_factor = 1 / scaler[0]
    # y_pred = y_pred * scale_factor
    # Y_test = Y_test * scale_factor


    # Signal
    def crossover(ma10, ma20, ma2):
        if ma10 > ma20 and ma20 > ma2:
            return 'buy'
        elif ma10 > ma20 or ma10 > ma2:
            return 'hold'
        else:
            return 'sell'


    ma10_recent = ma10.iloc[-1]
    ma20_recent = ma20.iloc[-1]
    ma200_recent = ma200.iloc[-1]
    signal = crossover(ma10_recent, ma20_recent, ma200_recent)

    # VIX
    close_prices = data['Close'].values
    log_returns = np.log(close_prices[1:] / close_prices[:-1])
    volatility_index = np.sqrt(252) * np.std(log_returns)

    threshold = 0.2

    if volatility_index < threshold:
        V_signal = 'buy'
    elif volatility_index > threshold:
        V_signal = 'sell'
    else:
        V_signal = 'hold'

    bullish_price_index = (close_prices[-1] - close_prices[0]) / close_prices[0]

    threshold_bullish = 0.03

    if bullish_price_index > threshold_bullish:
        b_signal = 'buy'
    elif bullish_price_index < -threshold_bullish:
        b_signal = 'sell'
    else:
        b_signal = 'hold'



    # Bollinger Bands
    close_prices = data['Close']
    mean = close_prices.rolling(window=20).mean()
    std = close_prices.rolling(window=20).std()

    # Upper and lower Bollinger Bands
    upper_band = mean + 2 * std
    lower_band = mean - 2 * std

    # Signal
    last_close_price = close_prices[-1]
    last_upper_band = upper_band[-1]
    last_lower_band = lower_band[-1]

    if last_close_price > last_upper_band:
        bol_signal = 'sell'
    elif last_close_price < last_lower_band:
        bol_signal = 'buy'
    else:
        bol_signal = 'hold'



    st.subheader('Sentiment Analysis')
    st.write("Overall Sentiment Score: ", overall_score)
    st.write("Buy/Sell Indication: ", buy_sell_indication)
    st.subheader('Technical Signal')
    st.write('Ema :', signal)
    st.write("Volatility Index(VIX):", V_signal)
    st.write("Bullish Price Index(BPI):", b_signal)
    st.write("Bollinger Bands:", bol_signal)