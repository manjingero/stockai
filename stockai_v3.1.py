import pandas as pd
import numpy as np
import requests
import tensorflow as tf
import json
import sys
from datetime import datetime, timedelta
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Version of application
VERSION = "3.1"

# API keys (user must place their own)
AV_API = "place your own API key"
NEWS_API = "place your own API key"

# Initiate the program
def main():
    print(f"Welcome to StockAI v{VERSION},")
    print("a Daniel Stone Product.\n")

    # Stocks to predict (user can switch this)
    stocks = ["TSLA", "AAPL", "AMZN", "SPY"] # you can add other ones

    for stock in stocks:
        # Retrieve data for the stock
        data = retrieveData(stock)

        # Convert data into LTSM model
        df, X_train, X_test, y_train, y_test = convertToLTSM(data)

        # Normalize the data using the MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)) # convert to 1 dimensinal array
        y_test = scaler.transform(y_test.values.reshape(-1, 1)) # convert to 1 dimensinal array

        # Create the neural network using TensorFlow
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8, input_dim = 8, activation = "relu"))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss = "mean_squared_error", optimizer = "adam") # I recommend sticking with the Adam optimizer

        # Train model
        model.fit(X_train, y_train, epochs = 99, batch_size = 32, verbose = 1)

        # Make prediction
        predictions = model.predict(X_test)

        # Transform the predictions back to the original scale (scaled down originally to make it faster)
        predictions = scaler.inverse_transform(predictions)

        # Get stock sentiment
        sentiment = getSentiment(stock)

        # Weigh data
        weighted_prediction = weighData(predictions, sentiment)

        # Show results
        showResults(df, weighted_prediction, stock)

        # Go on to the next stock
        input("Press ENTER to predict the next stock ")
    
def retrieveData(stock):
    # Load API
    url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={stock}&apikey={AV_API}")
    response = requests.get(url)

    # Return data
    return response.json()

def convertToLTSM(data):
    # Extract data
    df = pd.DataFrame(data["Time Series (Daily)"]).T
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace = True)

    # Convert to float (wasn't working as STR (shocking))
    df["Open"] = df["1. open"].astype(float)
    df["High"] = df["2. high"].astype(float)
    df["Low"] = df["3. low"].astype(float)
    df["Close"] = df["4. close"].astype(float)
    df["Adjusted_Close"] = df["5. adjusted close"].astype(float)
    df["Volume"] = df["6. volume"].astype(float)
    df["Dividend_Amount"] = df["7. dividend amount"].astype(float)
    df["Split_Coefficient"] = df["8. split coefficient"].astype(float)

    # Drop all STRs
    df = df.drop(["1. open", "2. high", "3. low", "4. close", "5. adjusted close", "6. volume", 
                  "7. dividend amount", "8. split coefficient"], axis = 1)

    # Shift the close price column by one day to create the target variable
    df["Target"] = df["Adjusted_Close"].shift(-1)

    # Split the data into train and test sets
    split_date = pd.Timestamp("2023-01-01")
    X_train, X_test = df[df.index < split_date], df[df.index >= split_date]
    y_train, y_test = X_train["Target"], X_test["Target"]

    # Convert the data into a format suitable for LSTM
    X_train = X_train.drop(["Target"], axis = 1)
    X_test = X_test.drop(["Target"], axis = 1)

    # Return all
    return df, X_train, X_test, y_train, y_test

def getSentiment(stock):
    try:
        # Get today's stock news
        # If no news today, it will retrieve ones up to a week old
        for i in range(7):
            today = datetime.today()
            day = today - timedelta(days = i)
            day = day.strftime("%Y-%m-%d")

            # Generate a valid URL
            url = f"https://newsapi.org/v2/everything?q={stock}&from={day}&sortBy=popularity&apiKey={NEWS_API}"

            # Record data
            response = requests.get(url)
            data = json.loads(response.text)

            stock_news_sentiment = []
            # Validate and retrieve sentiment
            if data["totalResults"] != 0:
                articles = data["articles"]
                for article in articles:
                    analysis = TextBlob(article["title"] + " " + article["description"])
                    polarity = analysis.sentiment.polarity
                    stock_news_sentiment.append(polarity)
                break
            else:
                continue
        
        # Get average of the list
        average = sum(stock_news_sentiment)/len(stock_news_sentiment)

        # Return average sentiment
        return average
    except:
        sys.exit("Error retrieving stock news.")

def weighData(predictions, sentiment):
    # Weigh predictions with sentiment
    predicted_price = float(predictions[-1])
    predicted_price *= (1+((sentiment - 0.1)/2))

    # Return new predicted value
    return predicted_price

def showResults(df, predicted_price, stock):
    # Show final results
    last_price = df['Close'][-1] # latest close number
    
    # Determine if predicted price is higher than last price
    if predicted_price > last_price:
        print(f"You should buy {stock}.")
    else:
        print(f"You should not buy {stock}.")

    # Show specific predicted results
    print(f"Price expected to go to: {round(predicted_price, 2)}, from {last_price}\n")

if __name__ == "__main__":
    main()