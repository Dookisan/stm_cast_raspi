import logging
import requests

logger = logging.getLogger(__name__)

class Requests:
    def __init__(self, SERVER_URL: str):
        self.SERVER_URL = SERVER_URL
        self.job_id = None
        self.filename = None
        self.files = None
        self.generated = []
        self.choice = None
        self.type = None  # to be set externally

    def upload_nn_models(self, choices):
        """Upload neural networks to the server"""
        logger.info("Uploading neural network models to the server...")

        if isinstance(choices, int):
            choice_list = [choices]
        elif isinstance(choices, slice):
            choice_list = list(range(choices.start or 0, choices.stop, choices.step or 1))
        elif isinstance(choices, range):
            choice_list = list(choices)
        else:
            choice_list = list(choices)
        self.choice = choice_list # saving for later compilation

        for choice in choice_list:
            logger.info(f"Uploading model for choice {choice}...")

            model = f"models/{self.type}_{choice}.tflite"

            with open(model, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.SERVER_URL}/upload", files=files)
        
            logger.debug(f"Status: {response.status_code}")
            result = response.json()
            logger.debug(f"Response: {result}")
    
        assert response.status_code == 200
        assert 'filename' in result
    
        logger.info("✅ PASSED")
        self.filename = result['filename']

    def generate_code(self, target: str, name: str):
        """Generate code for the uploaded model"""
        logger.info("Generating code for the uploaded model...")

        for choice in self.choice:
            logger.info(f"Generating code for model choice {choice}...")
            payload = {
                'filename': f"{self.type}_{choice}.tflite",
                'target':  target,
                'name': name
            }
            logger.debug(f"Payload: {payload}")
    
            response = requests.post(f"{self.SERVER_URL}/generate", json=payload)
    
            logger.debug(f"Status: {response.status_code}")
            result = response.json()
            logger.debug(f"Response:  {result}")

            self.generated.append(result["name"])
            logger.info(f"Generated code: {self.generated}")
        assert response.status_code == 200
        assert result['success'] == True
    
        logger.info("✅ PASSED")
        
 
    
    def list_outputs(self):
        """Test GET /outputs/{job_id}"""
        self.print_test(f"GET /outputs/{self.job_id}")
        
        response = requests.get(f"{self.SERVER_URL}/outputs/{self.job_id}")
    
        logger.debug(f"Status: {response.status_code}")
        result = response.json()
        logger.debug(f"Response: {result}")
    
        assert response.status_code == 200
        assert len(result['files']) > 0
    
        logger.info("✅ PASSED")
        self.files = result['files'][0]

    def set_type(self, type: str):
        self.type = type
        logger.info(f"Set model type to {self.type}")