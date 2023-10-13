"""Module to load trained model"""
from pathlib import Path
import tensorflow as tf
from src.car_detection import get_cd_config
from src.damage_detection import get_dd_config
from src.damage_severity import get_ds_config
from src import logger


class ModelsLoader:
    """Class to load models"""

    def __init__(self):
        pass

    def get_model_path(self, model_key) -> Path:
        """Method to get model path from config"""
        try:
            if model_key=="CD":
                config=get_cd_config()
            elif model_key=="DD":
                config=get_dd_config()
            elif model_key=="DS":
                config=get_ds_config()
            else:
                logger.exception("unknown model key")
                raise Exception

            model_config = config.get_model_config()
            return model_config.trained_model_path
        except AttributeError as ex:
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def load_model(self, model_key):
        """Method to load trained model"""
        try:
            model_path = self.get_model_path(model_key)
            model = tf.keras.models.load_model(model_path)
            logger.info("loaded %s model successfully.", model_key)
            return model
        except AttributeError as ex:
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
