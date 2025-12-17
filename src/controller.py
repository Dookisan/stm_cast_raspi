
from utils.data_processor import DataProcessor
from utils.plotter import Plotter
from src.deep_learning import NeuronalNetworkModel
from src.linear_corrector import FIRMultiStepPredictor
from utils.i2c import i2c_com
from api.requests import client

'''
Controller module for handling user requests and responses.

created_by: Elias Schebath
'''

class controller(object):
    def __init__(self):
        # Initialize controller components
        self.data_processor = None
        self.data_plotter = None
        self.neural_network_model = None
        self.fir_filter = None
        self.can_interrupt = False
        self.mongoose_data = None

    def current_status(self):
        '''
        Display current status of all components and data matrices
        '''
        print("\n" + "="*60)
        print("📊 CURRENT STATUS")
        print("="*60)

        # Check Data Processor
        print("\n🔧 Components Status:")
        print(f"  • Data Processor: {'✅ Loaded' if self.data_processor else '❌ Not loaded'}")
        print(f"  • Data Plotter: {'✅ Loaded' if self.data_plotter else '❌ Not loaded'}")
        print(f"  • FIR-Filter: {'✅ Loaded' if self.fir_filter else '❌ Not loaded'}")
        print(f"  • Neural Network Model: {'✅ Loaded' if self.neural_network_model else '❌ Not loaded'}")

        # Check Matrices
        print("\n📈 Data Matrices:")
        if self.data_processor and hasattr(self.data_processor, 'error_matrix'):
            if self.data_processor.error_matrix is not None:
                print(f"  • Error Matrix: ✅ Shape {self.data_processor.error_matrix.shape}")
            else:
                print(f"  • Error Matrix: ❌ Not generated")
        else:
            print(f"  • Error Matrix: ❌ Not available")

        if self.data_processor and hasattr(self.data_processor, 'error_target'):
            if self.data_processor.error_target is not None:
                print(f"  • Target Matrix: ✅ Shape {self.data_processor.error_target.shape}")
            else:
                print(f"  • Target Matrix: ❌ Not generated")
        else:
            print(f"  • Target Matrix: ❌ Not available")

        print("="*60 + "\n")

        if self.data_processor and hasattr(self.data_processor, 'error_matrix') and hasattr(self.data_processor, 'error_target'):
            answer = input("Do you want to check the current matrices? [y/n]: ")
            if answer.lower() == 'y':
                print("Origin Data:")
                print(self.data_processor.processed_data)
                print("Current Error Matrix:")
                print(self.data_processor.error_matrix)
                print("\nCurrent Target Matrix:")
                print(self.data_processor.error_target)
                print("="*60)

    #---------Data Processor Methods---------#
    def init_data_processor(self, filepath_station = None, filepath_weather = None):
        self.data_processor = DataProcessor(filepath_station, filepath_weather)

    def load_stm_data(self):
        if not self.data_processor:
            raise Exception("DataProcessor not initialized. Call init_data_processor first.")
        self.data_processor._load_stm_data()

    def load_data(self):
        if not self.data_processor:
            raise Exception("DataProcessor not initialized. Call init_data_processor first.")
        self.data_processor._load_data()

    def preprocess_data(self):
        if not self.data_processor:
            raise Exception("DataProcessor not initialized. Call init_data_processor first.")
        self.data_processor.preprocess_data()

    def get_day_error_sequence(self,day:int):
        if not self.data_processor:
            raise Exception("DataProcessor not initialized. Call init_data_processor first.")
        error_seq = self.data_processor.error_sequence_day(day)
        return error_seq

    def get_nn_input_layer(self, choice:int):
        if not self.data_processor:
            raise Exception("DataProcessor not initialized. Call init_data_processor first.")
        self.data_processor.nn_input_layer(choice)

    def get_fir_training_data(self,day_start:int, day_end:int):
        if not self.data_processor:
            raise Exception("DataProcessor not initialized. Call init_data_processor first.")
        self.data_processor.fir_training_input_sequence(day_start,day_end)

    def create_training_matrix(self):
        self.data_processor.create_corrector_training_maticies(bias_value = 1,lag=9,prediction_horizon = 24)

    def example_db_series(self):
        self.data_processor.example_db_series(self.client_api.mongoose_data)
    #---------Plotter Methods---------#

    def init_plotter(self):
        if self.data_processor.processed_data.empty:
            raise Exception("Data not preprocessed. Call preprocess_data first.")
        self.data_plotter = Plotter(self.data_processor.processed_data)

    def plot_time(self, prefix):
        if not self.data_plotter:
            raise Exception("Plotter not initialized. Call init_plotter first.")
        self.data_plotter.data_time(prefix)

    def plot_error(self):
        if not self.data_plotter:
            raise Exception("Plotter not initialized. Call init_plotter first.")
        self.data_plotter.error_plot()

    #---------Neuronal Network Methods---------#
    def init_neuronal_network(self):
        #TODO: Add init check for generate nn config
        self.neural_network_model = NeuronalNetworkModel()
        print("✅ Neuronal Network Model initialized.")

    def generate_nn_config(self):
        self.neural_network_model._init_config()
        print("✅ Neuronal Network configuration generated.")

    def split_matricies(self):
        if not self.data_processor:
            raise Exception("DataProcessor not initialized. Call init_data_processor first.")
        self.neural_network_model.train_test(self.data_processor.error_matrix,
                                             self.data_processor.error_target)

    def random_param_search(self):
        self.neural_network_model.run_simple_search()

    def analyze_model_results(self, top_n=5):
        self.neural_network_model.analyze_results()
        self.neural_network_model.show_top_configs(top_n=5)

    def save_NN_model(self, model_suffix=""):
        return self.neural_network_model.save_model(model_suffix=model_suffix)
    
    def train_multiple_models(self, choices):
        """
        Trainiert mehrere NN Modelle für verschiedene choice-Werte.
        
        Parameters:
        -----------
        choices : int, list, range, or slice
            - int: einzelner choice Wert
            - list/range: mehrere choice Werte [6, 12, 18]
            - slice: z.B. slice(6, 25, 6) für [6, 12, 18, 24]
        
        Returns:
        --------
        dict: Zusammenfassung aller trainierten Modelle
        """
        # Konvertiere Input zu Liste
        if isinstance(choices, int):
            choice_list = [choices]
        elif isinstance(choices, slice):
            choice_list = list(range(choices.start or 0, choices.stop, choices.step or 1))
        elif isinstance(choices, range):
            choice_list = list(choices)
        else:
            choice_list = list(choices)
        
        print("\n" + "="*60)
        print(f"🚀 MULTI-MODEL TRAINING")
        print("="*60)
        print(f"📊 Trainiere {len(choice_list)} Modelle für choices: {choice_list}")
        print("="*60 + "\n")
        
        results_summary = []
        
        for idx, choice in enumerate(choice_list, 1):
            print(f"\n{'='*60}")
            print(f"📦 MODELL {idx}/{len(choice_list)} - Choice={choice}")
            print("="*60)
            
            try:
                # 1. Input Layer erstellen
                self.get_nn_input_layer(choice=choice)
                
                # 2. NN konfigurieren (nur einmal bei erstem Modell)
                if idx == 1:
                    self.generate_nn_config()
                self.split_matricies()
                self.random_param_search()
                self.analyze_model_results(top_n=3)
                
                # 6. Modell speichern mit Suffix
                model_info = self.save_NN_model(model_suffix=str(choice))
                
                results_summary.append({
                    'choice': choice,
                    'model_number': idx,
                    'features': choice,
                    'samples': self.data_processor.error_matrix.shape[0],
                    'r2_score': model_info.get('r2'),
                    'h5_path': model_info.get('h5'),
                    'tflite_path': model_info.get('tflite'),
                    'status': 'success'
                })
                
                print(f"\n✅ Modell {idx} erfolgreich trainiert und gespeichert!")
                
            except Exception as e:
                print(f"\n❌ Fehler bei Modell {idx} (choice={choice}): {str(e)}")
                results_summary.append({
                    'choice': choice,
                    'model_number': idx,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Final Summary
        print("\n" + "="*60)
        print("📊 TRAINING ZUSAMMENFASSUNG")
        print("="*60)
        
        successful = [r for r in results_summary if r['status'] == 'success']
        failed = [r for r in results_summary if r['status'] == 'failed']
        
        print(f"✅ Erfolgreich: {len(successful)}/{len(choice_list)}")
        print(f"❌ Fehlgeschlagen: {len(failed)}/{len(choice_list)}")
        
        if successful:
            print("\n🏆 ERFOLGREICH TRAINIERTE MODELLE:")
            for r in successful:
                print(f"  • Choice={r['choice']:2d}: R²={r['r2_score']:.4f}, Samples={r['samples']:4d}")
                print(f"    Files: {r['h5_path']}")
        
        if failed:
            print("\n❌ FEHLGESCHLAGENE MODELLE:")
            for r in failed:
                print(f"  • Choice={r['choice']:2d}: {r['error']}")
        
        print("="*60 + "\n")
        
        return results_summary

    #------------Linear Model Methods------------------#

    def init_linear_model(self,lag:int):
        self.fir_filter = FIRMultiStepPredictor(lag)
        print("✅ FIR filter initialized.")

    def predict_linear_sequence(self):
        if not self.fir_filter:
            raise Exception("Initialize the linear Model first")
        self.fir_filter.fit(self.data_processor.E,self.data_processor.E_hat)
        #self.fir_filter.predict_sequence(self.data_processor.error_sequence, start_idx = 5)

    def wheater_prediction_examples(self):

        if not self.fir_filter:
            raise Exception("Initialize the linear Model first")
        self.fir_filter.predict_sequence(self.data_processor.ex, start_idx = 1)


    #---------I2C Methods---------#
    def init_i2c(self):
        self.i2c_com = i2c_com()
        print("✅ I2C communication initialized.")
    
    def check_can_interrupt(self):
        if not self.i2c_com:
            raise Exception("I2C communication not initialized. Call init_i2c first.")
        while(1):

            try:
                print(f"ich bin da")
                incoming = self.i2c_com.readArray()
                if 2 in  incoming:
                    print("⚠️ CAN Interrupt detected from STM32.")
                    self.can_interrupt = True
                    break
            except Exception as e:
                print(f"Fehler beim Lesen des I2C-Arrays: {e}")
            
    def write_i2c_array(self,data):
        if not self.i2c_com:
            raise Exception("I2C communication not initialized. Call init_i2c first.")
        elif not self.can_interrupt:
            raise Exception("No CAN interrupt detected. Cannot write data to I2C device.")  
        #self.data_processor.resize_to_can_frame(self.data_processor.prediction)

        try:
            self.i2c_com.writeArray(data)
            print("✅ Data array written to I2C device.")
        except Exception as e:
            print(f"Fehler beim Schreiben des I2C-Arrays: {e}")

    
    def write_i2c_byte(self):
        import time

        if not self.i2c_com:
            raise Exception("I2C communication not initialized. Call init_i2c first.")

        incoming = self.i2c_com.write_byte(self.mongoose_data.astype(int))
        #incoming = self.i2c_com.generateTelematry(self.data_processor.ex["weatherstation_temp"][0].astype(int))
        print(f"data i2c byte: {incoming}")
    """
    def write_i2c_array(self):
        if not self.i2c_com:
            raise Exception("I2C communication not initialized. Call init_i2c first.")
        elif not self.fir_filter.is_trained:
            raise Exception("FIR Model not trained. Call predict_linear_sequence first.")
        self.data_processor.resize_to_can_frame(self.data_processor.prediction)
        self.i2c_com.writeArray(self.data_processor.databyte)
        print("✅ Data array written to I2C device.")
      """
    #---------Client Methods---------#
    def init_client(self):
        self.client_api = client()
        print("✅ Client initialized.")

    def fetch_mongoose_data(self):
        if not self.client_api:
            raise Exception("Client not initialized. Call init_client first.")
        try:
            if not self.can_interrupt:
                raise Exception("No CAN interrupt detected. Cannot fetch data from Mongoose server.")
            self.client_api.get_data_mongoose()
            print("✅ Data fetched from Mongoose API.")
        except Exception as e:
            print(f"{e}")

    def preprocess_mongoose_data(self):
        if not self.client_api:
            raise Exception("Client not initialized. Call init_client first.")
        self.mongoose_data = self.data_processor.build_temp(self.client_api.mongoose_data)
        print("✅ Mongoose data preprocessed.")
    '''
    how i activatet them prevoisly

    # ===== BEISPIEL 1: Einzelnen FIR Predictor verwenden =====
print("=== Beispiel 1: Einzelner FIR Multi-Step Predictor ===")

# Testdaten erstellen
np.random.seed(42)
test_data = pd.Series(error_sequence_nonlinear(24, nonlin_scale=4))
print(f"Testdaten: {len(test_data)} Punkte")
print(f"test_data: \n {test_data}")
# FIR Predictor erstellen und trainierenpredict_sequence
predictor = FIRMultiStepPredictor(lag=4, prediction_horizon=6)
predictor.fit(test_data)

# Einzelne Vorhersage machen
prediction = predictor.predict_sequence(test_data, start_idx=5)
print(f"\\nVorhersage ab Index 5: {prediction.round(3)}")

# Modell-Informationen
info = predictor.get_info()
print(f"\\nModell-Info: {info}")

print("\\n" + "="*60 + "\\n")


or

# ===== BEISPIEL 2: Verschiedene Lag-Werte vergleichen =====
print("=== Beispiel 2: Lag-Vergleich mit Plotter ===")

# Plotter erstellen
plotter = FIRMultiStepPlotter()

# Verschiedene Lag-Werte vergleichen
rmse_results = plotter.plot_lag_comparison(
    series=test_data,
    prediction_horizon=6,
    lag_range=range(1, 8),  # Teste Lag 1 bis 7
    show_every_nth=2  # Zeige jede 2. Vorhersage
)

print("\\n" + "="*60 + "\\n")


or

# ===== BEISPIEL 3: Detaillierte Analyse eines Predictors =====
print("=== Beispiel 3: Detaillierte Analyse ===")

# Besten Lag aus vorherigem Vergleich verwenden (niedrigste RMSE)
best_lag = min(rmse_results, key=lambda x: x[1])[0]
print(f"Bester Lag (niedrigste RMSE): {best_lag}")

# Neuen Predictor mit bestem Lag trainieren
best_predictor = FIRMultiStepPredictor(lag=best_lag, prediction_horizon=6)
best_predictor.fit(test_data)

# Detaillierte Visualisierung
plotter.plot_single_predictor(test_data, best_predictor, max_predictions=4)

print("\\n" + "="*60 + "\\n")


or


# ===== BEISPIEL 4: One-Step vs Multi-Step Vergleich =====
print("=== Beispiel 4: One-Step vs Multi-Step Vergleich ===")

# Direkter Vergleich zwischen den Methoden
plotter.plot_comparison_one_vs_multi(
    series=test_data,
    lag=4,
    prediction_horizon=6
)

print("\\n" + "="*60)
print("🎉 Alle modularen Komponenten erfolgreich getestet!")
print("Die Komponenten sind jetzt bereit für Ihre Anwendung.")



with the last matrix of the

class FIRLagSweepPlot:
    def __init__(self, error_series, max_lag=None, nonlin_scale=7):
        self.error_series = error_series
        self.n = len(error_series)
        self.max_lag = max_lag if max_lag is not None else self.n
        self.nonlin_scale = nonlin_scale
        self.preds_lags = []
        self.rmses = []
        self.plot_handles = []
        self.plot_labels = []
        self._fit_all_lags()

    def _fit_all_lags(self):
        n = self.n
        for lag in range(1, self.max_lag):
            E, E_hat = [], []
            for i in range(n-lag):
                vals = self.error_series.iloc[i:i+lag][::-1].values
                in_vec = np.concatenate([vals, [1]])
                E.append(in_vec)
                E_hat.append(self.error_series.iloc[i+lag])
            E = np.array(E)
            E_hat = np.array(E_hat)
            if len(E) == 0 or len(E_hat) == 0:
                self.preds_lags.append(np.full(n, np.nan))
                self.rmses.append(np.nan)
                continue
            reg = LinearRegression(fit_intercept=False)
            reg.fit(E, E_hat)
            fir_coef = reg.coef_
            preds = np.full(n, np.nan)
            for t in range(lag, n):
                vals = self.error_series.iloc[t-lag:t][::-1].values
                in_vec = np.concatenate([vals, [1]])
                preds[t] = np.dot(in_vec, fir_coef)
            self.preds_lags.append(preds)
            mask = ~np.isnan(preds)
            if np.any(mask):
                rmse = np.sqrt(np.mean((self.error_series.values[mask] - preds[mask])**2))
                self.rmses.append(rmse)
            else:
                self.rmses.append(np.nan)

    def _get_color(self, lag_idx, n_lags):
        color_segments = [int(n_lags*0.33), int(n_lags*0.66), n_lags]
        if lag_idx < color_segments[0]:
            cmap = plt.get_cmap('Blues')
            return to_hex(cmap(0.4 + 0.6*lag_idx/color_segments[0]))
        elif lag_idx < color_segments[1]:
            cmap = plt.get_cmap('Greens')
            return to_hex(cmap(0.4 + 0.6*(lag_idx-color_segments[0])/(color_segments[1]-color_segments[0])))
        else:
            cmap = plt.get_cmap('autumn')
            return to_hex(cmap(0.2 + 0.8*(lag_idx-color_segments[1])/(color_segments[2]-color_segments[1])))

    def plot(self):
        n = self.n
        n_lags = self.max_lag-1
        fig = plt.figure(figsize=(16, 7))
        gs = GridSpec(1, 2, width_ratios=[1, 3], wspace=0.05)
        ax_legend = fig.add_subplot(gs[0])
        ax_plot = fig.add_subplot(gs[1])
        self.plot_handles = []
        self.plot_labels = []
        # Fehlerlinie
        ax_plot.plot(np.arange(n), self.error_series.values, color='black', linestyle='dashed', linewidth=2, label='Echte Fehler')
        for lag, preds in enumerate(self.preds_lags, start=1):
            color = self._get_color(lag-1, n_lags)
            handle, = ax_plot.plot(np.arange(n), preds, color=color, alpha=0.9)
            self.plot_handles.append(handle)
            self.plot_labels.append(f'FIR Prognose (lag={lag})')
            highlight_idx = lag if lag < n else None
            if highlight_idx is not None and not np.isnan(preds[highlight_idx]):
                ax_plot.scatter(highlight_idx, preds[highlight_idx], color='red', s=80, marker='o', edgecolor='black', zorder=10)
        ax_plot.set_xlabel('timestamp (hour)')
        ax_plot.set_ylabel('Fehler')
        ax_plot.set_title('FIR-Prognose für verschiedene lag-Werte (Highlight t=lag, Legende links)')
        ax_plot.grid(True)
        ax_legend.axis('off')
        ax_legend.legend(self.plot_handles, self.plot_labels, loc='center left', fontsize=9, frameon=False)
        plt.tight_layout()
        plt.show()
        # RMSE für jeden lag ausgeben
        for lag, rmse in enumerate(self.rmses, start=1):
            print(f'lag={lag}: RMSE={rmse:.4f}')


        n = 24
nonlin_scale = 7

np.random.seed(42)
errors_real = error_sequence_nonlinear(n, nonlin_scale)
error_series_real = pd.Series(errors_real, index=[f't={i}' for i in range(n)])

fir_lag_sweep = FIRLagSweepPlot(error_series_real, max_lag=n, nonlin_scale=nonlin_scale)
fir_lag_sweep.plot()

und den performance matriken


# Plotting der Trainingshistorie und Modell-Performance
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Loss History
training_loss = history.history["loss"]
val_loss = history.history["val_loss"]
epoch_count = range(1, len(training_loss) + 1)

axes[0,0].plot(epoch_count, training_loss, "r-", linewidth=2, label="Training Loss")
axes[0,0].plot(epoch_count, val_loss, "b-", linewidth=2, label="Validation Loss")
axes[0,0].set_xlabel("Epoch")
axes[0,0].set_ylabel("Loss (MSE)")
axes[0,0].set_title("Training vs Validation Loss")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. MAE History
training_mae = history.history["mae"]
val_mae = history.history["val_mae"]

axes[0,1].plot(epoch_count, training_mae, "r-", linewidth=2, label="Training MAE")
axes[0,1].plot(epoch_count, val_mae, "b-", linewidth=2, label="Validation MAE")
axes[0,1].set_xlabel("Epoch")
axes[0,1].set_ylabel("Mean Absolute Error")
axes[0,1].set_title("Training vs Validation MAE")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Vorhersagen vs echte Werte
y_pred = network.predict(X).flatten()
axes[1,0].scatter(y, y_pred, alpha=0.6, color='blue', s=50)
axes[1,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfekte Vorhersage')
axes[1,0].set_xlabel("Echte Werte")
axes[1,0].set_ylabel("Vorhergesagte Werte")
axes[1,0].set_title("Vorhersagen vs Echte Werte")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# R² Score berechnen und anzeigen
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
axes[1,0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1,0].transAxes,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
              fontsize=12, verticalalignment='top')

# 4. Residuen Plot
residuals = y - y_pred
axes[1,1].scatter(y_pred, residuals, alpha=0.6, color='green', s=50)
axes[1,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1,1].set_xlabel("Vorhergesagte Werte")
axes[1,1].set_ylabel("Residuen (Echt - Vorhersage)")
axes[1,1].set_title("Residuen Plot")
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistiken ausgeben
print("\n" + "="*50)
print("🧠 NEURAL NETWORK PERFORMANCE STATISTIKEN")
print("="*50)
print(f"📊 Finaler Training Loss:     {training_loss[-1]:.4f}")
print(f"📊 Finaler Validation Loss:   {val_loss[-1]:.4f}")
print(f"📊 Finaler Training MAE:      {training_mae[-1]:.4f}")
print(f"📊 Finaler Validation MAE:    {val_mae[-1]:.4f}")
print(f"📊 R² Score:                  {r2:.4f}")
print(f"📊 RMSE:                      {np.sqrt(np.mean(residuals**2)):.4f}")
print(f"📊 Anzahl Trainingssamples:   {len(X)}")
print(f"📊 Input Features:            {X.shape[1]}")
print("="*50)


    '''