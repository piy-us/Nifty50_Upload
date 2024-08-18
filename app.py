from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import math
import time
import matplotlib.pyplot as plt
import io
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

@app.route('/', methods=['GET', 'POST'])
def index():
    timestamp = int(time.time())
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        stock_time = '3y'
        number_of_days = int(request.form['number_of_days'])
        submit_btn = request.form['submit_btn']


        stock = yf.Ticker(stock_symbol)

        # Retrieve historical stock data
        historical_data = stock.history(period=stock_time)

        # Convert the data to a pandas DataFrame
        dataframe = pd.DataFrame(historical_data)

        # Process the data using the create_dataset function (time_step = 1 by default)
        df = dataframe.copy()
        if not df.empty:
            df1 = df.reset_index()["Close"]
            print(len(df1))
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

            # Load the pre-trained model (you need to have the model.h5 file in the same directory)
            # modname = "models4/"+stock_symbol + ".pkl"
            # with open(modname, 'rb') as f:
            #     model = pickle.load(f)
            if submit_btn == 'task1':
                modname = "models4/"+stock_symbol + ".pkl"
                with open(modname, 'rb') as f:
                    model = pickle.load(f)
            elif submit_btn=='task2':
                modname = "models/"+stock_symbol + ".pkl"
                with open(modname, 'rb') as f:
                    model = pickle.load(f)
                


            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            # Calculate RMSE performance metrics
            from sklearn.metrics import mean_squared_error

            train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
            test_rmse = math.sqrt(mean_squared_error(ytest, test_predict))

            # Convert the plot to an image and encode it to base64 for display on the webpage
            look_back = 100
            trainPredictPlot = np.empty_like(df1)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

            testPredictPlot = np.empty_like(df1)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

            fig = make_subplots()
            fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=scaler.inverse_transform(df1).flatten(), mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=trainPredictPlot.flatten(), mode='lines', name='Train Predicted', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=testPredictPlot.flatten(), mode='lines', name='Test Predicted', line=dict(color='green')))

            fig.update_layout(title='Actual vs Train vs Test Predictions',
                            title_font=dict(color='white'), 
                            xaxis=dict(title='Index'),
                            yaxis=dict(title='Close Price'),
                            legend=dict(font=dict(color='white')),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(color='white')
            fig.update_yaxes(color='white')

            # Save the plot to a static HTML file
            graph_filename1 = 'train_predictions.html'
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
            graph_filepath1 = os.path.join(static_dir, graph_filename1)
            fig.write_html(graph_filepath1)

            x_input = test_data[len(test_data)-100:].reshape(1, -1)
            temp_input = list(x_input)
            temp_input = temp_input[0].tolist()

            lst_output = []
            n_steps = 100
            i = 0
            while i < number_of_days:
                if len(temp_input) > 100:
                    x_input = np.array(temp_input[1:])
                    x_input = x_input.reshape(1, -1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=1)
                    temp_input.extend(yhat[0].tolist())
                    temp_input = temp_input[1:]
                    lst_output.extend(yhat.tolist())
                    i = i + 1
                else:
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=1)
                    temp_input.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i = i + 1

            # Prepare the output predictions for display on the webpage
            predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()
            #prediction_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=number_of_days)

            # Create a DataFrame to store the predictions and corresponding dates
            prediction_df = pd.DataFrame({'Predicted Close Price': predictions})

            # Save the DataFrame as a CSV file (optional)
            predictions_csv_filename = 'predictions.csv'
            #prediction_df.to_csv(predictions_csv_filename, index=False)
            print(len(df1))

           
            df3=df1.tolist()
            df3.extend(lst_output)
            df3=scaler.inverse_transform(df3).tolist()
            fig = make_subplots()
            fig.add_trace(go.Scatter(x=np.arange(len(df3)), y=scaler.inverse_transform(df3).flatten(), mode='lines', name='Actual', line=dict(color='blue')))
            fig.update_layout(title='Actual with Predictions Combined',
                              title_font=dict(color='white'), 
                              xaxis=dict(title='Index'),
                              yaxis=dict(title='Close Price'),
                              legend=dict(font=dict(color='white')),
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(color='white')
            fig.update_yaxes(color='white')
            graph_filename3 = 'df3.html'
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
            graph_filepath3 = os.path.join(static_dir, graph_filename3)
            fig.write_html(graph_filepath3)

            df3=df1.tolist()
            df3.extend(lst_output)
            df3=scaler.inverse_transform(df3).tolist()
    
            df4=df3[1:748]
            df5=df3[748:]

            day_new1 = np.arange(1, 748)
            day_pred1 = np.arange(748, 748 + number_of_days)
            fig = make_subplots()
            fig.add_trace(go.Scatter(x=day_new1, y=scaler.inverse_transform(df4).flatten(), mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=day_pred1, y=scaler.inverse_transform(df5).flatten(), mode='lines', name='Actual', line=dict(color='red')))

            fig.update_layout(title='Actual vs Predictions',
                              title_font=dict(color='white'), 
                              xaxis=dict(title='Index'),
                              yaxis=dict(title='Close Price'),
                              legend=dict(font=dict(color='white')),
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(color='white')
            fig.update_yaxes(color='white')
            graph_filename4 = 'df4.html'
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
            graph_filepath4 = os.path.join(static_dir, graph_filename4)
            fig.write_html(graph_filepath4)

            df3=df1.tolist()
            df3.extend(lst_output)
            df3=df3[500:]
            df3=scaler.inverse_transform(df3).tolist()
            fig = make_subplots()
            fig.add_trace(go.Scatter(x=np.arange(len(df3)), y=scaler.inverse_transform(df3).flatten(), mode='lines', name='Actual', line=dict(color='blue')))
            fig.update_layout(title='Actual with Prediction Zoom-In',
                              title_font=dict(color='white'), 
                              xaxis=dict(title='Index'),
                              yaxis=dict(title='Close Price'),
                              legend=dict(font=dict(color='white')),
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(color='white')
            fig.update_yaxes(color='white')
            graph_filename5 = 'df5.html'
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
            graph_filepath5 = os.path.join(static_dir, graph_filename5)
            fig.write_html(graph_filepath5)

            


            


            
                #############
                

            return render_template('index.html', train_rmse=train_rmse, test_rmse=test_rmse, plot_filename1=graph_filename1, number_of_days=number_of_days,plot_filename3=graph_filename3,plot_filename4=graph_filename4,plot_filename5=graph_filename5,timestamp=timestamp)

    return render_template('index.html')


            #return render_template('index.html', train_rmse=train_rmse, test_rmse=test_rmse, plot_filename1=graph_filename1, predictions=prediction_df.to_html(index=False), predictions_csv_filename=predictions_csv_filename, number_of_days=number_of_days)

    #return render_template('index.html')


            #return render_template('index.html', train_rmse=train_rmse, test_rmse=test_rmse, plot_filename1=graph_filename1)

    #return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
