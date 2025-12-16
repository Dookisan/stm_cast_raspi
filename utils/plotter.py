'''
This class handels all the plotting functionalities
and visualizations.

Author: Elias Schebath
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# nur für daweil bevor den fix
from src.linear_corrector import FIRMultiStepPredictor
class Plotter(object):
  def __init__(self,data,fir_init:str):
    self.data = data
    if fir_init.lower() == "y":
       fir_plot = FIRMultiStepPlotter()
       print(f"✅activateted FIRMSP in plotter")


  def initialize_plotting(self):
    if self.data is None:
      raise Exception("Data not provided for plotting.")
    print("✅ Plotter initialized with data.")

  def data_time(self,prefix):
    '''
    Plot value over time
    '''
    # Der Index ist ein MultiIndex, verwende einfach die Zeilennummer für x-Achse
    # oder erstelle einen zusammengesetzten Zeitstempel
    plt.figure(figsize=(12,6))

    # Verwende den Index direkt (MultiIndex mit month, day, hour)
    x_values = range(len(self.data))

    plt.plot(x_values, self.data[f'{prefix}_stm'], label='STM Temperature', marker='o')
    plt.plot(x_values, self.data[f'{prefix}_api'], label='API Temperature', marker='x')
    plt.xlabel('Observation Index')
    plt.ylabel(f'{prefix}')
    plt.title(f'{prefix} over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

  def error_plot(self):
    '''
    Plot error between STM and API temperature readings
    '''
    error = self.data['error']
    plt.figure(figsize=(12,6))
    x_values = range(len(self.data))

    plt.plot(x_values, error, label='Temperature Error (STM - API)', color='red', marker='d')
    plt.xlabel('Observation Index')
    plt.ylabel('Temperature Error')
    plt.title('Temperature Error over Time')
    plt.axhline(0, color='black', linestyle='--')
    plt.show()


class FIRMultiStepPlotter:
    """
    Visualisierungs-Komponente für FIR Multi-Step Vorhersagen

    Diese Klasse erstellt verschiedene Plots um FIR Multi-Step Vorhersagen
    zu visualisieren und zu vergleichen.
    """

    def __init__(self):
        """Initialisiert den Plotter mit Standard-Einstellungen"""
        # Bessere Farbpalette für verschiedene Lags
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Standard Plot-Einstellungen
        self.figsize = (16, 8)
        self.original_linewidth = 3
        self.prediction_linewidth = 2.5
        self.legend_linewidth = 3

    def plot_lag_comparison(self, series: pd.Series, prediction_horizon: int,
                           lag_range: range, show_every_nth: int = 2):
        """
        Vergleicht Multi-Step Vorhersagen für verschiedene Lag-Werte

        Args:
            series: Ursprüngliche Zeitreihe
            prediction_horizon: Anzahl vorherzusagender Schritte
            lag_range: Range der zu testenden Lag-Werte (z.B. range(1, 9))
            show_every_nth: Zeigt jede n-te Vorhersage (für Übersichtlichkeit)
        """
        n = len(series)

        plt.figure(figsize=self.figsize)

        # Original Zeitreihe plotten
        plt.plot(range(n), series.values, 'ko-',
                linewidth=self.original_linewidth, markersize=7,
                label='Original Zeitreihe', zorder=10)

        rmse_results = []

        for i, lag in enumerate(lag_range):
            # FIR Predictor für diesen Lag trainieren
            predictor = FIRMultiStepPredictor(lag=lag, prediction_horizon=prediction_horizon)

            try:
                predictor.fit(series)
            except ValueError as e:
                print(f"Warnung für Lag {lag}: {e}")
                continue

            # Vorhersagen für alle möglichen Startpunkte
            predictions = []
            start_points = []

            for start_idx in range(0, n - prediction_horizon + 1):
                try:
                    pred_seq = predictor.predict_sequence(series, start_idx)
                    predictions.append(pred_seq)
                    start_points.append(start_idx)
                except (ValueError, IndexError):
                    continue

            # Vorhersagen plotten (nur jede n-te für Übersichtlichkeit)
            color = self.colors[i % len(self.colors)]

            for j, (start_idx, pred_seq) in enumerate(zip(start_points, predictions)):
                if j % show_every_nth == 0:
                    pred_indices = range(start_idx, start_idx + prediction_horizon)
                    alpha = 0.9 if j == 0 else 0.7  # Erste Vorhersage hervorheben
                    plt.plot(pred_indices, pred_seq, color=color,
                            alpha=alpha, linewidth=self.prediction_linewidth,
                            linestyle='--')

            # RMSE für Legende
            rmse = predictor.get_info()['rmse']
            rmse_results.append((lag, rmse))

            # Legende-Eintrag
            plt.plot([], [], color=color, linewidth=self.legend_linewidth,
                    linestyle='--', label=f'Lag {lag} (RMSE: {rmse:.3f})')

        # Plot finalisieren
        plt.xlabel('Zeitpunkt', fontsize=12)
        plt.ylabel('Wert', fontsize=12)
        plt.title(f'FIR Multi-Step-Ahead Vergleich (Horizon = {prediction_horizon})',
                 fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # RMSE Tabelle ausgeben
        print("\\nRMSE Ergebnisse:")
        print("-" * 25)
        for lag, rmse in rmse_results:
            print(f"Lag {lag:2d}: RMSE = {rmse:.4f}")

        return rmse_results

    def plot_single_predictor(self, series: pd.Series, predictor: FIRMultiStepPredictor,
                             max_predictions: int = 5):
        """
        Zeigt detaillierte Vorhersagen für einen einzelnen FIR Predictor

        Args:
            series: Ursprüngliche Zeitreihe
            predictor: Trainierter FIRMultiStepPredictor
            max_predictions: Maximale Anzahl von Vorhersage-Sequenzen zu zeigen
        """
        if not predictor.is_trained:
            print("Fehler: Predictor muss erst trainiert werden!")
            return

        n = len(series)
        info = predictor.get_info()

        plt.figure(figsize=self.figsize)

        # Original Zeitreihe
        plt.plot(range(n), series.values, 'ko-',
                linewidth=self.original_linewidth, markersize=7,
                label='Original Zeitreihe', zorder=10)

        # Vorhersagen für gleichmäßig verteilte Startpunkte
        max_start = n - info['prediction_horizon']
        step = max(1, max_start // max_predictions)
        colors_pred = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, start_idx in enumerate(range(0, max_start + 1, step)):
            if i >= max_predictions:
                break

            pred_seq = predictor.predict_sequence(series, start_idx)
            pred_indices = range(start_idx, start_idx + info['prediction_horizon'])

            color = colors_pred[i % len(colors_pred)]
            plt.plot(pred_indices, pred_seq, '--', color=color,
                    linewidth=3, alpha=0.8,
                    label=f'Vorhersage ab t={start_idx}')

        plt.xlabel('Zeitpunkt', fontsize=12)
        plt.ylabel('Wert', fontsize=12)
        plt.title(f'FIR Multi-Step Vorhersagen (Lag={info["lag"]}, '
                 f'Horizon={info["prediction_horizon"]}, RMSE={info["rmse"]:.3f})',
                 fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Modell-Informationen ausgeben
        print(f"\\nModell-Informationen:")
        print(f"- Lag (Filterlänge): {info['lag']}")
        print(f"- Prediction Horizon: {info['prediction_horizon']}")
        print(f"- Trainingsbeispiele: {info['n_training_samples']}")
        print(f"- Input Features: {info['input_features']}")
        print(f"- RMSE: {info['rmse']:.4f}")
        print(f"- Koeffizienten-Matrix: {info['coefficients_shape']}")

    def plot_comparison_one_vs_multi(self, series: pd.Series, lag: int,
                                   prediction_horizon: int):
        """
        Vergleicht One-Step vs Multi-Step Vorhersagen direkt

        Args:
            series: Zeitreihe
            lag: Lag-Wert für beide Methoden
            prediction_horizon: Horizon für Multi-Step
        """
        n = len(series)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # === ONE-STEP-AHEAD ===
        ax1.plot(range(n), series.values, 'ko-', linewidth=2, markersize=6,
                label='Original', zorder=10)

        # One-Step implementieren (wie FIR Lag Sweep)
        E_one, E_hat_one = [], []
        for i in range(n-lag):
            vals = series.iloc[i:i+lag][::-1].values
            in_vec = np.concatenate([vals, [1]])
            E_one.append(in_vec)
            E_hat_one.append(series.iloc[i+lag])

        E_one = np.array(E_one)
        E_hat_one = np.array(E_hat_one)

        reg_one = LinearRegression(fit_intercept=False)
        reg_one.fit(E_one, E_hat_one)

        preds_one = np.full(n, np.nan)
        for t in range(lag, n):
            vals = series.iloc[t-lag:t][::-1].values
            in_vec = np.concatenate([vals, [1]])
            preds_one[t] = reg_one.predict([in_vec])[0]

        ax1.plot(range(n), preds_one, 'r--', linewidth=2, alpha=0.8,
                label=f'One-Step (lag={lag})')
        ax1.set_title('One-Step-Ahead')
        ax1.set_xlabel('Zeit')
        ax1.set_ylabel('Wert')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # === MULTI-STEP-AHEAD ===
        ax2.plot(range(n), series.values, 'ko-', linewidth=2, markersize=6,
                label='Original', zorder=10)

        # Multi-Step Predictor verwenden
        predictor = FIRMultiStepPredictor(lag=lag, prediction_horizon=prediction_horizon)
        predictor.fit(series)

        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, start_idx in enumerate(range(0, n - prediction_horizon + 1, 4)):
            #if i >= 5:  # Maximal 5 Vorhersagen zeigen
            #    break
            pred_seq = predictor.predict_sequence(series, start_idx)
            pred_indices = range(start_idx, start_idx + prediction_horizon)

            color = colors[i % len(colors)]
            ax2.plot(pred_indices, pred_seq, '--', color=color, linewidth=2,
                    alpha=0.8, label=f'Multi-Step t={start_idx}')

        ax2.set_title(f'Multi-Step-Ahead (Horizon={prediction_horizon})')
        ax2.set_xlabel('Zeit')
        ax2.set_ylabel('Wert')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\\nVergleich:")
        print(f"One-Step:   Jeder Zeitpunkt einzeln vorhergesagt")
        print(f"Multi-Step: Sequenzen von {prediction_horizon} Werten auf einmal")
        print(f"Multi-Step RMSE: {predictor.get_info()['rmse']:.4f}")