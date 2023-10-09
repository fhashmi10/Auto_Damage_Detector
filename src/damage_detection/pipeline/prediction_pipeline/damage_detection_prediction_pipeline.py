"""Module to create model prediction pipeline"""
from src.damage_detection.configuration.damage_detection_configuration_manager \
    import DamageDetectionConfigurationManager
from src.damage_detection.components.model.model_predictor import ModelPredictor
from src import logger

def damage_detection_prediction_pipeline(filename):
    """Method to perform prediction"""
    try:
        config = DamageDetectionConfigurationManager()
        model_predictor = ModelPredictor(data_config=config.get_data_config(),
                                         model_config=config.get_model_config(),
                                         params=config.get_param_config())
        prediction = model_predictor.predict(filename)
        return prediction
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
