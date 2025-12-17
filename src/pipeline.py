
"""
Custom Pipeline for controller handeling
Will be used for automate workflows which should not be
as felxible as in the user interface

created by: Elias Schebath
"""

class Pipeline(object):

    def __init__(self,name: str, controller):
        self.controller = controller
        self.name = name
        self.steps = []

    def add_step(self, method_name: str, *args, **kwargs):
        self.steps.append((method_name, args, kwargs))
        return self
    
    def run(self):
        for method_name, args, kwargs in self.steps:
            method = getattr(self.controller, method_name)
            method(*args, **kwargs)
        return self.controller
    

class PipelineBuilder(object):

    @staticmethod
    def multi_mododel_training_pipeline(controller):
        """creates all nn for a 24h forecast"""
        

    @staticmethod
    def single_model_pipeline(controller):
        """trains a single nn model for given choice"""
        pipeline = Pipeline("model_training",controller)

        (pipeline.add_step("init_data_processor", filepath_station="data/Weatherstation_STMCAST_Data_202511102036.csv",
                        filepath_weather="data/Weather_API_Data_202511102036.csv")
        .add_step("load_data")
        .add_step("preprocess_data")
        .add_step("init_neuronal_network")
        .add_step("get_nn_input_layer", 6)
        .add_step("generate_nn_config")
        .add_step("split_matricies")
        .add_step("random_param_search")
        .add_step("analyze_model_results")
        )
        return pipeline

    @staticmethod
    def linear_model_pipeline(controller):
        """traines the linear model for 24h forecast"""
        pipeline = Pipeline("fir_forecast", controller)

        (pipeline.add_step("init_data_processor", filepath_station="data/Weatherstation_STMCAST_Data_202511102036.csv",
                        filepath_weather="data/Weather_API_Data_202511102036.csv")
        .add_step("load_data")
        .add_step("preprocess_data")
        .add_step("init_linear_model",9)
        .add_step("get_fir_training_data",21,24)
        .add_step("create_training_matrix")
        .add_step("predict_linear_sequence")
        )
        return pipeline
    
    @staticmethod
    def mongoose_pipeline(controller):
        """fetches mongoose data and returns it to i2c"""

        pipeline = Pipeline("mongoose_i2c", controller)
        (pipeline.add_step("init_i2c")
        .add_step("init_client")
        .add_step("check_can_interrupt")
        .add_step("fetch_mongoose_data")
        .add_step("preprocess_mongoose_data")
        .add_step("write_i2c_array")
        )
        return pipeline

    @staticmethod
    def simulation_pipeline(controller):
        """simulates the error sequence for day 18 and writes it to i2c"""
        
        pipeline = Pipeline("simulation", controller)
        (pipeline.add_step("init_i2c")
        .add_step("init_client")
        .add_step("check_can_interrupt")
        .add_step("get_day_error_sequence", day = 18)
        .add_step("write_i2c_array")
        )
        return pipeline
