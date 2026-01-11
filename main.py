from __future__ import annotations
from src.pipeline import Pipeline, PipelineBuilder
from src.controller import controller
    


def main():
    print("Starting STM-Cast")
    the_builder = PipelineBuilder()

    # Automatic pipelines for the continuse forecasting system

    """
    fir = the_builder.linear_model_pipeline(controller())
    fir.run()

    nn = the_builder.single_model_pipeline(controller()) # no automatic save
    nn.run()
    
    simulation = the_builder.simulation_pipeline(controller())
    simulation.run()
    
    nn_training_pipeline = the_builder.multi_mododel_training_pipeline(controller())
    nn_training_pipeline.run()

    cli = the_builder.client_pipeline(controller())
    cli.run()
    
    print("\n🎯 Fertig! Alle Modelle sind im 'models/' Ordner gespeichert.")


    
    while True:
     # === MONGOOSE COMMUNICATION PIPELINE ===      
        server = the_builder.mongoose_pipeline(controller())
        server.run()
        
    """
    
    

     # === MULTI-MODEL TRAINING ===
    # Option 1: Einzelnes Modell
    # CONTROLLER.train_multiple_models(6)
    
    # Option 2: Liste von Choices
    #CONTROLLER.train_multiple_models([6, 12, 18])
    
    # Option 3: Range
    # CONTROLLER.train_multiple_models(range(6, 25, 6))  # [6, 12, 18, 24]
    
    # Option 4: Slice 
   # results = CONTROLLER.train_multiple_models(slice(1, 22, 1))  # [6, 12, 18, 24]

    # === CONTINUES RUN  === 

    cli = the_builder.client_pipeline(controller())
    cli.run()
    
    

if __name__ == "__main__":
    main()
