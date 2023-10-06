"""Module to load trained model"""
import tensorflow as tf
from src.car_detection.configuration.car_detection_configuration_manager \
    import CarDetectionConfigurationManager
from src import logger

class ModelsLoader:
    """Class to load models"""

    def __init__(self):
        pass

    def load_car_detection_model(self):
        """Method to load trained model"""
        try:
            config = CarDetectionConfigurationManager()
            model_config=config.get_model_config()
            model =  tf.keras.models.load_model(model_config.trained_model_path)
            logger.info("loaded model successfully.")
            return model
        except AttributeError as ex:
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
