import time
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from src.models.lstm_models import LSTMModelFactory
from src.utils.metrics import calculate_metrics, format_metrics
from src.utils.config import load_config
from src.utils.visualization import plot_results
from src.training.callbacks import ProgressMonitor

class ModelTrainer:
    def __init__(self):
        self.config = load_config()

    def train_model(self, 
                   train_X, train_y, 
                   test_X, test_y,
                   scaler,
                   stock_name,
                   scenario ="w_Sentiment_and_Keywords",
                   model_type="simple_lstm",
                   verbose=1):

        if verbose:
            print(f"Training {model_type} model for {stock_name} - {scenario}")
            print(f"Input shape: {train_X.shape[1:]}")
            print(f"Training samples: {train_X.shape[0]}, Test samples: {test_X.shape[0]}")

        # Create model
        model = LSTMModelFactory.create_model(
            input_shape=(train_X.shape[1], train_X.shape[2]),
            model_type=model_type,
            units=self.config['model']['units'],
            dropout_rate=self.config['model']['dropout_rate'],
            activation=self.config['model']['activation']
        )

        if verbose:
            model.summary()

        # Compile model
        model.compile(
            optimizer=self.config['training']['optimizer']['name'],
            loss='mean_squared_error'
        )

        # Setup callbacks
        callbacks = []
        callbacks.extend([
            ProgressMonitor(epochs=self.config['training']['epochs']),
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping']['patience'],
                restore_best_weights=True
            )
        ])

        # Train model
        start_time = time.time()
        history = model.fit(
            train_X, train_y,
            validation_split=0.1,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks,
            verbose=verbose
        )
        training_time = time.time() - start_time

        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds.")

        # Make predictions
        predictions = model.predict(test_X, verbose=0)

        # Inverse transform predictions and actual values
        predictions_concat = np.concatenate(
            [np.zeros((predictions.shape[0], scaler.n_features_in_ - 1)), predictions],
            axis=1
        )
        predictions_original = scaler.inverse_transform(predictions_concat)[:, -1]

        test_y_reshaped = test_y.reshape(-1, 1)
        test_y_concat = np.concatenate(
            [np.zeros((test_y_reshaped.shape[0], scaler.n_features_in_ - 1)), test_y_reshaped],
            axis=1
        )
        test_y_original = scaler.inverse_transform(test_y_concat)[:, -1]

        # Calculate metrics
        metrics = calculate_metrics(
            actual = test_y,
            prediction = predictions,
            training_time=training_time
        )
        if verbose: 
            format_metrics(metrics=metrics)
        
        if verbose: 
            plots = plot_results(
                stock_name= stock_name,
                history=history,
                actual = test_y_original,
                prediction=predictions_original
            )

        return model, history, metrics