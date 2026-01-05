import logging

logger = logging.getLogger(__name__)

class Requests:
    def __init__(self, SERVER_URL: str):
        self.SERVER_URL = SERVER_URL
        self.job_id = None
        self.filename = None
        self.files = None

    def upload_nn_models(self):
        """Upload neural networks to the server"""
        logger.info("Uploading neural network models to the server...")

        model = "./models/best_model_1.tflite"
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
