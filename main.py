from __future__ import annotations
from src.controller import controller

def server(CONTROLLER: controller):
    CONTROLLER.init_i2c()
    CONTROLLER.init_client()
    while(1):
        CONTROLLER.check_can_interrupt()
        CONTROLLER.fetch_mongoose_data()
        CONTROLLER.preprocess_mongoose_data()
        CONTROLLER.write_i2c_array()

def simulate(CONTROLLER: controller):
    CONTROLLER.init_i2c()
    CONTROLLER.init_client()
    while(1):
        CONTROLLER.check_can_interrupt()
        err = CONTROLLER.get_day_error_sequence(day = 18)
        
        CONTROLLER.write_i2c_array(err)   

def checkfunction():
        pass
    
def main():
    print("Starting STM-Cast")
    CONTROLLER = controller()
    CONTROLLER.init_data_processor("data/Weatherstation_STMCAST_Data_202511102036.csv",
                                   "data/Weather_API_Data_202511102036.csv")
    CONTROLLER.load_data()
    CONTROLLER.preprocess_data()
    #CONTROLLER.init_plotter()
    #CONTROLLER.plot_time("temperature")
    #CONTROLLER.plot_error()
    #CONTROLLER.init_neuronal_network()
    #CONTROLLER.get_nn_input_layer(choice=6)
    #CONTROLLER.generate_nn_config()
    #CONTROLLER.init_linear_model(lag = 9 )
    #CONTROLLER.get_day_error_sequence(day = 18) #Aufpassen Tag 18 fängt es an
    #CONTROLLER.get_fir_training_data(21,24)
    #CONTROLLER.create_training_matrix()
    #CONTROLLER.predict_linear_sequence()
    simulate(CONTROLLER)
    
        
    #CONTROLLER.example_db_series()
    #CONTROLLER.wheater_prediction_examples()
    
    #CONTROLLER.write_i2c_byte()
    #CONTROLLER.current_status()
    #CONTROLLER.split_matricies()
    #CONTROLLER.random_param_search()
    #CONTROLLER.analyze_model_results(top_n=5)
    #CONTROLLER.save_NN_model()

     # === MULTI-MODEL TRAINING ===
    # Option 1: Einzelnes Modell
    # CONTROLLER.train_multiple_models(6)
    
    # Option 2: Liste von Choices
    #CONTROLLER.train_multiple_models([6, 12, 18])
    
    # Option 3: Range
    # CONTROLLER.train_multiple_models(range(6, 25, 6))  # [6, 12, 18, 24]
    
    # Option 4: Slice (empfohlen)
   # results = CONTROLLER.train_multiple_models(slice(1, 22, 1))  # [6, 12, 18, 24]
    
    print("\n🎯 Fertig! Alle Modelle sind im 'models/' Ordner gespeichert.")

if __name__ == "__main__":
    main()
