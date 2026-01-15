import logging
from client.utils.logging_config import setup_logging
from client.discovery import discover_server, heartbeat
from client.utils.test_suite import TestSuite
from client.api_requests import Requests
import sys
import requests
import pandas as pd

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
        self.Server_URL = discover_server()
        heartbeat(self.Server_URL)
        self.requests = Requests(self.Server_URL) 

        self.reques_mongoose = "http://10.0.0.42/json_data" 
        self.mongoose_data = None

    def upload_neural_networks(self, choices):
        """Uploads neural networks to the server"""
        self.requests.upload_nn_models(choices)

    def generate_code(self, target: str, name: str):
        """Generates code for the uploaded models"""
        self.requests.generate_code(target, name)
    
    def get_data_mongoose(self):
        response = requests.get(self.reques_mongoose)
        df = pd.read_json(response.text, lines=True)
        self.mongoose_data = df

    def update_database(self):
        """Updates the database with the latest data from the mongoose webserver"""
        database_data =  requests.get(self.reques_mongoose, timeout=5)

        if database_data.status_code == 200:
            data = database_data.json()
            logger.debug(f"Database data: {data}")
        else:
            logger.error("Error:", database_data.status_code)

        return data
    
# testfunction for development
def main():
    CLIENT = Client()
    Tests = TestSuite(CLIENT.Server_URL)
    if input("Press 1 to start tests...") == "1":
        Tests.server_test()
        pass

    REQUESTS = Requests(CLIENT.Server_URL)
    REQUESTS.upload_nn_models(range(1, 24, 1))
    REQUESTS.generate_code(target="stm32f4", name="my_model")

    
if __name__ == '__main__': 
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    log_level = logging.DEBUG if verbose else logging.INFO
    
    setup_logging(level=log_level)
    main()