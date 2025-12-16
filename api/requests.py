import requests
import pandas as pd

class client():
    def __init__(self):
        self.reques_mongoose = "http://10.0.0.42/json_data" 
        self.mongoose_data = None

    def get_data_mongoose(self):
        response = requests.get(self.reques_mongoose)
        df = pd.read_json(self.reques_mongoose,lines = True)
        self.mongoose_data = df