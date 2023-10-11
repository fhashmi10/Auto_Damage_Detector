"""Module to create model prediction pipeline"""
from src.damage_severity.configuration.damage_severity_configuration_manager \
    import DamageSeverityConfigurationManager
from src.damage_severity.components.model.model_predictor import ModelPredictor
from src import logger

def damage_severity_prediction_pipeline(filename):
    """Method to perform prediction"""
    try:
        config = DamageSeverityConfigurationManager()
        model_predictor = ModelPredictor(data_config=config.get_data_config(),
                                         model_config=config.get_model_config(),
                                         params=config.get_param_config())
        prediction = model_predictor.predict(filename)
        return prediction
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
