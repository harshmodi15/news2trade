import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import json
from pathlib import Path
import os
from src.utils.visualization import plot_results

class Predictor:

    def prepare_data(stock_name: str,
                    input_data: pd.DataFrame,
                    input_date: str,
                    window_size: int = 10
                    )-> Tuple(pd.DataFrame, int):
                    
    def predict(stock_name: str, 
                input_data, 
                input_date, 
                trained_model_path: str  = "../pretrained_model")-> Tuple(str):

        trained_model_path = Path(f"{trained_model_path}/{stock_name}")
        config_path = f'{trained_model_path}/{stock_name}/{stock_name}_Sentiment-Keyword-LSTM_config.json'
        model_path = f'{trained_model_path}/{stock_name}/{stock_name}_Sentiment-Keyword-LSTM_best_model.keras'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        window_size = config['training_params']['window_size']
        
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                        'sentiment_score', 'Keyword_impact']
        target_column = 'Close'
        
        # find input date index
        input_date_idx = input_data[input_data['Date'] == input_date].index[0]
            
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        X = feature_scaler.fit_transform(input_data[feature_columns])
        y = target_scaler.fit_transform(input_data[[target_column]])
        
        # create input sequence
        last_sequence = X[input_date_idx-window_size:input_date_idx] 
        current_sequence = np.reshape(last_sequence, (1, window_size, len(feature_columns)))

        # load model
        model = tf.keras.models.load_model(model_path)
        
        # predict
        predictions = []
        dates = []
        current_date = datetime.strptime(input_date, '%Y-%m-%d')

        for i in range(prediction_length):
            pred = model.predict(current_sequence, verbose=0)
            pred_price = target_scaler.inverse_transform(pred)[0][0]
            predictions.append(pred_price)
            dates.append(current_date.strftime('%Y-%m-%d'))
            
            # Update sequence for next prediction
            if i < prediction_length - 1:
                if input_date_idx + i < len(X) - 1:
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1] = X[input_date_idx + i] 
                    
                    # Move to next date
                    current_date += timedelta(days=1)
        
        predictions = np.array(predictions)

        print(f"\nPrediction for {stock_name} on {input_date}")
        print("=" * 30)
        print(f"{'Date':<12}{'Predicted':>10}")
        for i,(date, pred) in enumerate(zip(dates, predictions)):
            print(f"{date:<12} {pred:>10.2f} ")
        