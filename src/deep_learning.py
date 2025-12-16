import random
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# TensorFlow/Keras Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Scikit-learn Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

'''
Creates and traines Neuronal Network configurations
Created by: Elias Schebath
'''
class NeuronalNetworkModel(object):
    def __init__(self):
      self.param_space = None
      self.config = None

    def _init_config(self):
        self.param_space = {
                      'optimizer': ['adam', 'rmsprop', 'nadam'],
                      'epochs': [50, 100, 150, 200],
                      'batch_size': [2, 4, 8, 16],
                      'hidden_units': [16, 32, 64, 128],
                      'activation': ['relu', 'tanh'],
                      'dropout': [0.0, 0.1, 0.2, 0.3],
                      'learning_rate': [0.001, 0.01, 0.1],
                      'hidden_layers': [1, 2],
                      'patience': [10, 15, 20]
                    }
        self.config = None

    def generate_config(self):
        """Generiert zufällige Konfiguration."""
        config = {}

        for param, values in self.param_space.items():
            config[param] = random.choice(values)
        self.config = config

        print(f"\n✅ SETUP KOMPLETT!")
        print(f"🎯 Parameter Space: {len(self.param_space)} Parameter definiert")
        print(f"📊 Daten bereit für Neural Network Training")


    def train_test(self,error_matrix, error_target):
        # DATEN SETUP - Train/Test Split
        print(f"\n📊 DATEN VORBEREITUNG:")
        print("="*60 + "")
        print(f"Original Data: X{error_matrix.shape}, y{error_target.shape}")

        # Train-Test Split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            error_matrix, error_target,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        print(f"Training Set: X{self.X_train.shape}, y{self.y_train.shape}")
        print(f"Test Set: X{self.X_test.shape}, y{self.y_test.shape}")
        print("="*60 + "\n")
        print(f"✅ Training kann gestartet werden")


    def _test_config(self,config, test_num):
        """
        Initialisiert das NN mit den random selectierten parametern.
        Wird von random search immer wieder aufgerufen.
        """
        print(f"\n🧪 Test #{test_num}")
        print(f"Config: {config}")

        try:
            # Model erstellen
            model = Sequential()
            model.add(Dense(config['hidden_units'], activation=config['activation'],
                        input_shape=(self.X_train.shape[1],)))

            if config['dropout'] > 0:
                model.add(Dropout(config['dropout']))

            # Extra Hidden Layers
            for i in range(config['hidden_layers'] - 1):
                units = max(config['hidden_units'] // 2, 8)
                model.add(Dense(units, activation=config['activation']))

            # Output Layer
            model.add(Dense(1))

            # Compile
            if config['optimizer'] == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
            else:
                optimizer = config['optimizer']

            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            # Training mit Early Stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=config['patience'],
                                    restore_best_weights=True, verbose=0)

            history = model.fit(self.X_train, self.y_train,
                            epochs=config['epochs'],
                            batch_size=config['batch_size'],
                            validation_split=0.2,
                            verbose=0,
                            callbacks=[early_stop])

            # Evaluation
            y_pred = model.predict(self.X_test, verbose=0)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

            actual_epochs = len(history.history['loss'])

            print(f"✅ R² = {r2:.4f}, RMSE = {rmse:.4f}, Epochs = {actual_epochs}")

            return {
                'test_number': test_num,
                'config': config,
                'r2_score': r2,
                'rmse': rmse,
                'actual_epochs': actual_epochs,
                'success': True,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }

        except Exception as e:
            print(f"❌ Fehler: {str(e)}")
            return {
                'test_number': test_num,
                'config': config,
                'r2_score': -999,
                'error': str(e),
                'success': False
            }

    def run_simple_search(self,target_tests=30, batch_size=5):
        """
        run_simple_search ist das Herzstück der NNM Klasse. Sie sucht
        mit zufälligen Einstellungen um Zeit zu sparen NN Konfigurations.
        Es werden immer 5 Konfigurationen getestet (batch size) für eine
        maximale Anzahl an test um vorzeitig einschreiten zu können.
        ---------------------------------------------------------------

        Wenn die run_simple_search die configs ausgewählt hat initialisiert
        die test_config methode das NN

        """
        self.search_results = []

        print(f"🚀 STARTE SIMPLE RANDOM SEARCH")
        print(f"🎯 Ziel: {target_tests} Tests in {batch_size}er Batches")
        print("="*50)

        current_tests = len(self.search_results)

        for batch_start in range(current_tests, target_tests, batch_size):
            batch_end = min(batch_start + batch_size, target_tests)

            print(f"\n📦 BATCH: Tests {batch_start + 1} bis {batch_end}")
            print("-" * 30)

            for test_num in range(batch_start + 1, batch_end + 1):
                # Generiere zufällige Config
                config = self.generate_config()

                # Teste die Config
                result = self._test_config(self.config, test_num)

                if result:
                    self.search_results.append(result)

                    # Early Stop bei excellentem Ergebnis
                    if result.get('r2_score', -999) > 0.7:
                        print(f"\n🎉 EXCELLENT R² = {result['r2_score']:.4f}!")
                        print("🏆 Early Stop - Ziel erreicht!")
                        return len(self.search_results)

            # Batch Summary
            if self.search_results:
                valid_results = [r for r in self.search_results if r['success']]
                if valid_results:
                    best_r2 = max(r['r2_score'] for r in valid_results)
                    avg_r2 = np.mean([r['r2_score'] for r in valid_results])
                    print(f"\n📊 Batch Summary: Best R² = {best_r2:.4f}, Avg = {avg_r2:.4f}")


        print(f"\n🏁 SEARCH ABGESCHLOSSEN!")
        print(f"✅ {len(self.search_results)} Tests durchgeführt")
        return len(self.search_results)

    def save_model(self, model_suffix=""):
        suffix = f"_{model_suffix}" if model_suffix else ""
        print(f"\n💾 SPEICHERE MODEL RESULTS{suffix}")
        print("="*40)
        print(f"🏆 Trainiere finales Modell mit bester Config...")

        # Baue Modell mit bester Config
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.config['hidden_units'], activation=self.config['activation'], input_shape=(self.X_train.shape[1],)))
        for _ in range(self.config['hidden_layers'] - 1):
            model.add(tf.keras.layers.Dense(self.config['hidden_units'], activation=self.config['activation']))
            model.add(tf.keras.layers.Dropout(self.config['dropout']))
        model.add(tf.keras.layers.Dense(1))

        # Kompiliere und trainiere
        model.compile(optimizer=self.config['optimizer'], loss='mse', metrics=['mae'])
        model.fit(self.X_train, self.y_train, epochs=self.config['epochs'], batch_size=self.config['batch_size'],
                  validation_data=(self.X_test, self.y_test), verbose=0)

        # Speichere Modell im Keras 3 Format
        h5_path = f'models/best_model{suffix}.keras'  # .keras statt .h5!
        model.save(h5_path)
        print(f"✅ Modell gespeichert: {h5_path}")

        # Konvertiere zu TensorFlow Lite
        print(f"🔄 Konvertiere zu TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # KEINE Quantisierung für bessere Genauigkeit
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # ❌ Auskommentiert!
        tflite_model = converter.convert()

        # Speichere TFLite Modell
        tflite_path = f'models/best_model{suffix}.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite Modell gespeichert: {tflite_path}")
        print(f"📊 Größe: {len(tflite_model) / 1024:.2f} KB")
        
        return {'h5': h5_path, 'tflite': tflite_path, 'r2': self.get_best_r2()}
    
    def get_best_r2(self):
        """Gibt den besten R² Score zurück"""
        valid_results = [r for r in self.search_results if r['success']]
        if valid_results:
            return max(r['r2_score'] for r in valid_results)
        return None

    def analyze_results(self):
        """Einfache Analyse der Ergebnisse."""
        if not self.search_results:
            print("⚠️ Keine Ergebnisse vorhanden")
            return

        valid_results = [r for r in self.search_results if r['success']]
        failed_results = [r for r in self.search_results if not r['success']]

        print("📊 RESULTS ANALYSIS")
        print("=" * 40)
        print(f"📈 Total Tests: {len(self.search_results)}")
        print(f"✅ Successful: {len(valid_results)}")
        print(f"❌ Failed: {len(failed_results)}")

        if valid_results:
            r2_scores = [r['r2_score'] for r in valid_results]
            best = max(valid_results, key=lambda x: x['r2_score'])

            print(f"\n🏆 BEST RESULT:")
            print(f"   R² Score: {best['r2_score']:.4f}")
            print(f"   RMSE: {best['rmse']:.4f}")
            print(f"   Config: {best['config']}")

            print(f"\n📊 STATISTICS:")
            print(f"   Mean R²: {np.mean(r2_scores):.4f}")
            print(f"   Median R²: {np.median(r2_scores):.4f}")
            print(f"   Min/Max: {np.min(r2_scores):.4f} / {np.max(r2_scores):.4f}")

    def show_top_configs(self,top_n=3):
        """Zeigt die besten Konfigurationen."""
        valid_results = [r for r in self.search_results if r['success']]

        if not valid_results:
            print("⚠️ Keine erfolgreichen Tests vorhanden")
            return

        # Sortiere nach R² Score
        sorted_results = sorted(valid_results, key=lambda x: x['r2_score'], reverse=True)

        print(f"🏆 TOP {min(top_n, len(sorted_results))} KONFIGURATIONEN")
        print("=" * 60)

        for i, result in enumerate(sorted_results[:top_n]):
            self.config = result['config']
            print(f"\n#{i+1}: R² = {result['r2_score']:.4f}")
            print(f"   Hidden: {self.config['hidden_units']}, Optimizer: {self.config['optimizer']}")
            print(f"   LR: {self.config['learning_rate']}, Epochs: {result['actual_epochs']}")

        print("🚀 Simple Search Functions bereit!")
        print("\n💡 VERWENDUNG:")
        print("   run_simple_search()     # Startet mit 30 Tests")
        print("   analyze_results()       # Zeigt Analyse")
        print("   show_top_configs()      # Zeigt Top 3")