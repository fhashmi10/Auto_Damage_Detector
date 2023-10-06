"""Module to map data from config to dataclasses"""
from src.car_detection.configuration import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml_configbox
from src.car_detection.entities.config_entity import DataConfig, ModelConfig, \
    ParamConfig, CallbackConfig, EvaluationConfig
from src import logger


class ConfigurationManager:
    """Class to manage configuration"""

    def __init__(self):
        try:
            self.config = read_yaml_configbox(CONFIG_FILE_PATH)
            self.params = read_yaml_configbox(PARAMS_FILE_PATH)
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def get_data_config(self) -> DataConfig:
        """Method to manage data configuration"""
        try:
            config = self.config.data
            data_config = DataConfig(source_url=config.source_url,
                                     data_path=config.data_path,
                                     class_labels_path=config.class_labels_path)
            return data_config
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def get_model_config(self) -> ModelConfig:
        """Method to manage model configuration"""
        try:
            config = self.config.model
            model_config = ModelConfig(model_url=config.model_url,
                                       base_model_path=config.base_model_path,
                                       built_model_path=config.built_model_path,
                                       transform_model_path=config.transform_model_path,
                                       trained_model_path=config.trained_model_path)
            return model_config
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def get_param_config(self) -> ParamConfig:
        """Method to manage param configuration"""
        try:
            params = self.params
            param_config = ParamConfig(trainable=params.trainable,
                                       augmentation=params.augmentation,
                                       image_size=params.image_size,
                                       batch_size=params.batch_size,
                                       number_classes=params.number_classes,
                                       number_epochs=params.number_epochs,
                                       learning_rate=params.learning_rate,
                                       dropout_rate=params.dropout_rate,
                                       l2_pentaly_rate=params.l2_pentaly_rate)
            return param_config
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def get_callback_config(self) -> CallbackConfig:
        """Method to manage call back configuration"""
        try:
            config = self.config.callback
            callback_config = CallbackConfig(callback_path=config.callback_path,
                                             # tensorboard_log_path=config.tensorboard_log_path,
                                             model_checkpoint_path=config.model_checkpoint_path)
            return callback_config
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex

    def get_evaluation_config(self) -> EvaluationConfig:
        """Method to manage evaluation configuration"""
        try:
            config = self.config.eval
            track_params = {}
            for param in config.track_params.split(','):
                track_params[param] = self.params.get(param, None)

            eval_config = EvaluationConfig(
                test_data_path=config.test_data_path,
                evaluation_score_json_path=config.evaluation_score_json_path,
                mlflow_uri=config.mlflow_uri,
                track_params=track_params)
            return eval_config
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
