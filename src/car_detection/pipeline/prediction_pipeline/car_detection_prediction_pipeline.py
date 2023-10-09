"""Module to create model prediction pipeline"""
from src.car_detection.configuration.car_detection_configuration_manager \
    import CarDetectionConfigurationManager
from src.car_detection.components.model.model_predictor import ModelPredictor
from src import logger

def car_detection_prediction_pipeline(filename):
    """Method to perform prediction"""
    try:
        config = CarDetectionConfigurationManager()
        model_predictor = ModelPredictor(data_config=config.get_data_config(),
                                         model_config=config.get_model_config(),
                                         params=config.get_param_config())
        prediction = model_predictor.predict(filename)
        return prediction
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
