"""Module to create model prediction pipeline"""
from src.configuration.configuration_manager import ConfigurationManager
from src.car_detection.components.model.cd_model_predictor import CarDetectionPredictor
from src import logger


class CarDetectionPredictionPipeline:
    """Class to create model prediction pipeline"""

    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        """Method to perform prediction"""
        try:
            config = ConfigurationManager()
            model_predictor = CarDetectionPredictor(config=config.get_car_detection_config())
            prediction = model_predictor.predict(self.filename)
            return [{"image": prediction}]
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
