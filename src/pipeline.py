
from utils.data_processor import DataProcessor
from utils.plotter import Plotter
from src.deep_learning import NeuronalNetworkModel
from src.linear_corrector import FIRMultiStepPredictor
from utils.i2c import i2c_com
from api.requests import client

"""
created_by: Elias Schebath
Base pipeline model for easier processing in the controller class
"""

class Pipeline:
    def __init__(self):
        self.steps = []

    def add(self, func):
        self.steps.append(func)
        return self

    def run(self):
        for func in self.steps:
            data = func()
        return data
    
write_pipeline = Pipeline()
write_pipeline.add()