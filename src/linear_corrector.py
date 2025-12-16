import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

'''
Acts as the FIR Model of the STM Cast weather prediction.

Creator: Schebath Elias
'''

class FIRMultiStepPredictor:
    """
    Sie kann mit verschiedenen Lag-Werten trainiert werden und mehrere zukünftige
    Werte auf einmal vorhersagen.
    """

    def __init__(self, lag: int, bias_value: float = 1.0):
        """
        Initialisiert den FIR Multi-Step Predictor

        Args:
            lag: Anzahl vergangener Werte als Input (Filterlänge)
            prediction_horizon: Anzahl zukünftiger Werte vorherzusagen
            bias_value: Bias-Wert für Padding und als zusätzliches Feature
        """
        self.lag = lag
        self.bias_value = bias_value
        self.regressor = None
        self.is_trained = False

        self.prediction = None
        self.rmse = None

    def fit(self, training_matrix_E,training_matrix_E_hat):
        """
        Trainiert den FIR-Filter auf der gegebenen Zeitreihe

        Args:
            series: Trainingszeitreihe als pandas Series
        """

        # Multi-Output Linear Regression trainieren
        self.regressor = LinearRegression(fit_intercept=False)
        self.regressor.fit(training_matrix_E, training_matrix_E_hat)

        # RMSE berechnen für den theoretischen Wert
        self.predictions = self.regressor.predict(training_matrix_E)
        self.rmse = np.sqrt(np.mean((training_matrix_E_hat - self.predictions)**2))

        self.is_trained = True

        print(f"✓ FIR Multi-Step Predictor trainiert:")
        print(f"  - Lag: {self.lag}")
        print(f"  - Trainingsbeispiele: {len(training_matrix_E)}")
        print(f"  - RMSE: {self.rmse:.4f}")
        print(f"a matrix: {self.predictions}")

    def predict_sequence(self, series: pd.Series, start_idx: int):
        """
        Macht eine Multi-Step Vorhersage ab einem bestimmten Startindex

        Args:
            series: Zeitreihe für den Input-Kontext
            start_idx: Startindex für die Vorhersage

        Returns:
            numpy.array: Vorhersage-Sequenz der Länge prediction_horizon
        """
        series = pd.Series(series)
        if not self.is_trained:
            raise ValueError("Modell muss erst mit fit() trainiert werden!")

        # Input-Vektor für Startpunkt erstellen
        if start_idx < self.lag:
            pad = [self.bias_value] * (self.lag - start_idx)
            vals = series.iloc[:start_idx][::-1].values
            in_vec = np.concatenate([vals, pad])
        else:
            vals = series.iloc[start_idx-self.lag:start_idx][::-1].values
            in_vec = vals

        in_vec = np.concatenate([in_vec, [self.bias_value]])

        # Vorhersage machen
        prediction = self.regressor.predict([in_vec])[0]
        self.prediction = prediction
    
'''
    def get_info(self):
        """
        Gibt Informationen über den trainierten Predictor zurück

        Returns:
            dict: Informationen über das Modell
        """
        return {
            'lag': self.lag,
            'prediction_horizon': self.prediction_horizon,
            'is_trained': self.is_trained,
            'rmse': self.rmse,
            'n_training_samples': len(self.training_matrix_E) if self.is_trained else 0,
            'input_features': self.lag + 1,  # lag + bias
            'coefficients_shape': self.regressor.coef_.shape if self.is_trained else None
        }
'''


