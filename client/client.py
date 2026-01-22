import logging
from client.utils.logging_config import setup_logging
from client.discovery import discover_server, heartbeat
from client.utils.test_suite import TestSuite
from client.api_requests import Requests
import sys
import requests
import pandas as pd
import time
import json

# Configure logging
setup_logging(
    level=logging.INFO,
    log_file='logs/server.log',
    console=True
)

# Get logger for this module
logger = logging.getLogger(__name__)


class Client(object):

    def __init__(self):
        
        self.reques_mongoose = "http://192.168.4.2/json_data" 
        self.mongoose_data = None
        self.predictions_temp = "http://192.168.4.2/weather_prediction_data_temp"
        self.predictions_hum = "http://192.168.4.2/weather_prediction_data_hum"

        self.data_training = self.get_data_training()
        self.data_pred_temp = self.get_temperature_predictions()
        self.data_pred_hum = self.get_humidity_predictions()
        self.requests = None
        
    def init_remote_updater(self):
        self.Server_URL = discover_server()
        heartbeat(self.Server_URL)
        self.requests = Requests(self.Server_URL) 

    def upload_neural_networks(self, choices):
        """Uploads neural networks to the server"""
        self.requests.upload_nn_models(choices)

    def generate_code(self, target: str, name: str):
        """Generates code for the uploaded models"""
        self.requests.generate_code(target, name)
    
    def get_data_mongoose(self):
        while True:
            try:
                response = requests.get(self.reques_mongoose, timeout=5)
                df = pd.read_json(response.text, lines=True)
                self.mongoose_data = df
                break
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching mongoose data: {e}")
                time.sleep(5)   
                  

    def update_database(self, max_retries=None, retry_delay=5):
        """Updates the database with the latest data from the mongoose webserver
        
        Args:
            max_retries: Maximum number of retry attempts. None = infinite retries
            retry_delay: Seconds to wait between retries (default: 5)
        
        Returns:
            dict: JSON data from mongoose webserver
        """
        attempt = 0
        
        while True:
            attempt += 1
            try:
                if max_retries is not None:
                    logger.info(f"Attempting to fetch data from mongoose (attempt {attempt}/{max_retries})...")
                else:
                    logger.info(f"Attempting to fetch data from mongoose (attempt {attempt})...")
                
                database_data = requests.get(self.reques_mongoose, timeout=5)
                
                if database_data.status_code == 200:
                    data = database_data.json()
                    logger.info(f"✅ Successfully fetched data from mongoose")
                    logger.debug(f"Database data: {data}")
                    return data
                else:
                    logger.error(f"Error: HTTP {database_data.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching data from mongoose: {e}")
            
            # Check if we should retry
            if max_retries is not None and attempt >= max_retries:
                logger.error(f"Max retries ({max_retries}) reached. Giving up.")
                return None
            
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    def get_data_training(self):
        while True:
            try:
                response = requests.get(self.reques_mongoose,timeout =10)
                data = response.json()
                return data
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching training data: {e}")
                time.sleep(10) 

    def get_temperature_predictions(self):
        while True:
            try:
                response = requests.get(self.predictions_temp, timeout=10)
                data = response.json()
                return data
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching temperature predictions: {e}")
                time.sleep(10)
    
    def get_humidity_predictions(self):
        while True:
            try:
                response = requests.get(self.predictions_hum, timeout=10)
                data = response.json()
                return data
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching humidity predictions: {e}")
                time.sleep(10)