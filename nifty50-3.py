import os
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Function to create a dataset for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Create a folder to store the pickle files
output_folder = "models3"

os.makedirs(output_folder, exist_ok=True)

stock_symbols = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TATACONSUM.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
]

# Text input for the stock name
for stock_name in stock_symbols:
    stock_time = '3y'

    stock = yf.Ticker(stock_name)

    # Retrieve historical stock data for 3 years
    historical_data = stock.history(period=stock_time)

    # Convert the data to a pandas DataFrame
    dataframe = pd.DataFrame(historical_data)

    # Display the DataFrame
    df = dataframe

    if not df.empty:
        df1 = df.reset_index()["Close"]
        epochs_str = 100
        batch_size_str = 64

        epochs = int(epochs_str)
        batch_size = int(batch_size_str)
        scaler = MinMaxScaler(feature_range=(0, 1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

        training_size = int(len(df1) * 0.70)
        test_size = len(df1) - training_size
        train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

        time_step = 100  # You can change this value if needed

        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], time_step, 1)
        X_test = X_test.reshape(X_test.shape[0], time_step, 1)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # model = Sequential()
        # model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        # model.add(Dropout(0.3))
        # model.add(LSTM(units=64, return_sequences=True))
        # model.add(Dropout(0.3))
        # model.add(LSTM(units=64, return_sequences=True))
        # model.add(Dropout(0.3))
        # model.add(LSTM(units=64))
        # model.add(Dropout(0.3))
        # model.add(Dense(units=1))
        # optimizer = Adam(lr=0.001)
        # model.compile(optimizer=optimizer, loss='mean_squared_error')

        # model = Sequential()
        # model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        # model.add(Dropout(0.3))
        # model.add(LSTM(units=64, return_sequences=True))
        # model.add(Dropout(0.3))
        # model.add(LSTM(units=64))
        # model.add(Dropout(0.3))
        # model.add(Dense(units=1))
        # optimizer = Adam(lr=0.001)
        # model.compile(optimizer=optimizer, loss='mean_squared_error')
        # model = Sequential()
        # model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        # model.add(LSTM(units=50))
        # model.add(Dense(units=1))
        # model.compile(optimizer='adam', loss='mean_squared_error')
        # from keras.layers import Bidirectional

        # model = Sequential()
        # model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
        # model.add(Bidirectional(LSTM(units=50)))
        # model.add(Dense(units=1))
        # model.compile(optimizer='adam', loss='mean_squared_error')





        # Fit the model with user-specified parameters
        history = model.fit(X_train, y_train, validation_data=(X_test, ytest),
                            epochs=epochs, batch_size=batch_size, verbose=1)

        # Save the model using the stock name as the file name in the output folder
        model_filename = os.path.join(output_folder, f"{stock_name}.pkl")
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
