"""Module to create model prediction pipeline"""
from src.image_classification.configuration.configuration_manager import ConfigurationManager
from src.image_classification.components.model.model_predictor import ModelPredictor
from src import logger

def image_classification_prediction(config_file_path, params_file_path, filename):
    """Method to perform prediction"""
    try:
        config = ConfigurationManager(config_file_path=config_file_path,
                                          params_file_path=params_file_path)
        model_predictor = ModelPredictor(data_config=config.get_data_config(),
                                         model_config=config.get_model_config(),
                                         params=config.get_param_config())
        prediction = model_predictor.predict(filename)
        return prediction
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
