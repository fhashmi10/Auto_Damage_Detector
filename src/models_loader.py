"""Module to load trained model"""
import tensorflow as tf
from src.car_detection.configuration.car_detection_configuration_manager \
    import CarDetectionConfigurationManager
from src.damage_detection.configuration.damage_detection_configuration_manager \
    import DamageDetectionConfigurationManager
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
            logger.info("loaded car detection model successfully.")
            return model
        except AttributeError as ex:
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)

    def load_damage_detection_model(self):
        """Method to load trained model"""
        try:
            config = DamageDetectionConfigurationManager()
            model_config=config.get_model_config()
            model =  tf.keras.models.load_model(model_config.trained_model_path)
            logger.info("loaded damage detection model successfully.")
            return model
        except AttributeError as ex:
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
