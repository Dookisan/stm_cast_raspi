from unittest import result
from matplotlib.pylab import choice
import pandas as pd
import numpy as np


'''
Data processing module for handling and transforming data.
created_by: Elias Schebath
'''
class DataProcessor(object):

  def __init__(self,filepath_stm = None,filepath_api = None):
    # Initialize DataProcessor with file paths
    self.filepath_stm = filepath_stm
    self.filepath_api = filepath_api

    self.weather_stm = None
    self.weather_api = None

    self.processed_data = None

    self.error_matrix = None
    self.error_target = None

    self.error_sequence = None
    self.E = None
    self.E_hat = None

    self.ex = None
    self.predictions = None
    self.databyte = None

  def _load_data(self):
    '''
    Load the given data files on the system path.
    **Developer Note**: It has no current error handling for file not found nor
    incorrect file format.
    Returns:
    '''
    stm = pd.read_csv(self.filepath_stm, sep=';')
    vals = ['temperature','pressure','humidity']
    stm = stm.replace(',', '.', regex=True).set_index('ID')
    weather_stm = stm[vals].apply(pd.to_numeric, errors='coerce').assign(observation_time=pd.to_datetime(stm['observation_time']))

    api = pd.read_csv(self.filepath_api, sep=';').set_index('ID')
    api = api.replace(',', '.', regex=True)
    weather_api = api[vals].apply(pd.to_numeric, errors='coerce').assign(observation_time=pd.to_datetime(api['observation_time']))
    print(f"✅ Data loaded successfully. Preprocessing can begin now.")

    self.weather_stm = weather_stm
    self.weather_api = weather_api
    # TODO: Add error handling and return functions

  def _load_stm_data(self):
      stm = pd.read_csv(self.filepath_stm, sep=';')
      vals = ['temperature','pressure','humidity']
      stm = stm.replace(',', '.', regex=True).set_index('ID')
      weather_stm = stm[vals].apply(pd.to_numeric, errors='coerce').assign(observation_time=pd.to_datetime(stm['observation_time']))
      self.weather_stm = weather_stm

  def preprocess_data(self):

    '''
    Preprocess Data in one Database

    Returns:
      pd.DataFrame: Preprocessed DataFrame
    '''

    #TODO: Make it into a pipeline with multiple steps
    self.weather_stm = self.weather_stm.reset_index(drop=True)
    self.weather_api = self.weather_api.reset_index(drop=True)

    self.weather_stm = self.weather_stm.dropna(subset=['observation_time'])
    self.weather_api = self.weather_api.dropna(subset=['observation_time'])

   # Beide DataFrames sortieren
    self.weather_stm = self.weather_stm.sort_values('observation_time').reset_index(drop=True)
    self.weather_api = self.weather_api.sort_values('observation_time').reset_index(drop=True)

   # Fuzzy Join: Zu jedem Wert aus df1 wird der jeweils nächste (innerhalb ±10min) aus df2 genommen
    result = pd.merge_asof(
       self.weather_stm,
       self.weather_api,
    on='observation_time',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=20),
    suffixes=('_stm', '_api')
    )

    result = result.dropna().reset_index(drop=True)
    error = result['temperature_stm'] - result['temperature_api']
    result = result.assign(error=error)
    result = result.set_index('observation_time')
    head = result.columns.values

    easy_access = pd.MultiIndex.from_arrays([result.index.month,result.index.day, result.index.hour], names = ['month','day','hour'])

    subsets = pd.DataFrame(result.values,index = easy_access,columns = head)
    self.processed_data = subsets
    print(f"✅ Data successfully preprocessed.")

  def error_sequence_day(self,day):
    self.error_sequence = self.processed_data["error"].loc[self.processed_data.index.get_level_values('day') == day]
    print(f"✅ Error sequence of a day created successfully.")
    return self.error_sequence


  def nn_input_layer(self,choice:int):
    '''
    Create feature matrix and target vector for given hour choice
    FIXED: Verwendet nur ECHTE Features, keine Bias-Füllung mehr!
    
    Parameters:
    --------------
    dataset : DataFrame
          The full dataset with multiindex
    choice : int
          Anzahl der vergangenen Stunden als Features (z.B. 6 = Stunden 0-5)
          Target ist dann Stunde (choice)
    '''

    def get_subset(subset, time):
      return subset.loc[subset.index.get_level_values('hour') == time]

    # find minimum shape for all subsets to align the error matrix
    sub_shapes = []
    for idx in range(0, choice):
      subset = get_subset(self.processed_data, idx)
      sub_shapes.append(subset.shape[0])
    
    # Target auch berücksichtigen
    target_subset = get_subset(self.processed_data, choice)
    sub_shapes.append(target_subset.shape[0])
    
    min_shape = min(sub_shapes)
    
    print(f"📊 NN Input Layer Erstellung:")
    print(f"   Features: Stunden 0-{choice-1} ({choice} Features)")
    print(f"   Target: Stunde {choice}")
    print(f"   Anzahl Samples: {min_shape}")

    # Erstelle Feature Matrix OHNE Bias-Füllung
    feature_columns = []
    for past in range(0, choice):
      error_subset = get_subset(self.processed_data, past)
      error_subset = error_subset.tail(min_shape)
      feature_columns.append(error_subset['error'].values)
    
    # Stack alle Features horizontal
    error_matrix = np.column_stack(feature_columns)
    
    # Target: Fehler zur Stunde 'choice'
    error_target = get_subset(self.processed_data, choice)['error'].tail(min_shape).values

    print(f"   ✅ Feature Matrix Shape: {error_matrix.shape}")
    print(f"   ✅ Target Vector Shape: {error_target.shape}")
    print(f"   ✅ Keine Bias-Füllung - nur echte Features!")

    self.error_matrix = error_matrix
    self.error_target = error_target

  def fir_training_input_sequence(self,day_start,day_end, bring_me_the_prediction_horizon = 24):

    day_values = self.processed_data["error"].index.get_level_values('day')
    day_range = range(day_start, day_end)  # [1, 2, 3, 4]
    mask = day_values.isin(day_range)

    result = self.processed_data["error"].loc[mask]
    self.error_sequence = result

  def create_corrector_training_maticies(self,bias_value,lag, prediction_horizon):
        """
        Erstellt die Trainingsmatrizen E (Input) und E_hat (Output)

        Args:
            series: Eingangszeitreihe als pandas Series

        Returns:
            tuple: (E_matrix, E_hat_matrix) als numpy arrays
        """
        series = self.error_sequence

        n_samples = len(series)
        n_training_samples = prediction_horizon + lag

        if n_training_samples > n_samples:
            raise ValueError(f"Nicht genug Daten: {n_samples} Samples, "
                           f"aber {prediction_horizon} prediction_horizon benötigt")

        E, E_hat = [], []

        for i in range(lag + 1):
            # Input-Vektor erstellen: vergangene 'lag' Werte + Bias
            if i < lag:
                # Am Anfang: mit Bias-Werten auffüllen
                pad = [bias_value] * (lag - i)
                vals = series.iloc[:i][::-1].values  # Umkehren für FIR-Konvention
                in_vec = np.concatenate([vals, pad])
            else:
                # Normale Operation: lag vergangene Werte
                vals = series.iloc[i-lag:i][::-1].values
                in_vec = vals

            # Bias-Term hinzufügen
            in_vec = np.concatenate([in_vec, [bias_value]])
            E.append(in_vec)

            # Output-Vektor: nächste 'prediction_horizon' Werte
            future_values = series.iloc[i:i+prediction_horizon].values
            E_hat.append(future_values)

        self.E = np.array(E)
        self.E_hat = np.array(E_hat)

  def example_db_series(self,mongoose_data):
      err = mongoose_data["weatherstation_temp"] - mongoose_data["temperature"]
      mongoose_data = mongoose_data.assign(error=err)
      self.ex = mongoose_data



  def resize_to_can_frame(self,prediction):
    self.databyte =  np.array(prediction,dtype=np.uint8)
      
  
  def build_temp(self,mongoose_data):
      temp = mongoose_data.fillna(0)
      value = temp.values
      return value
     