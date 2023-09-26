"""Module to create model builder pipeline"""
from src.car_detection.configuration.configuration_manager import ConfigurationManager
from src.car_detection.components.model.model_builder import ModelBuilder
from src import logger

def model_builder_pipeline():
    """Method to perform model building"""
    try:
        stage_name = "Car Detection Model Build"
        logger.info("%s started", stage_name)
        config = ConfigurationManager()
        model_builder = ModelBuilder(config=config.get_model_config(),
                                     params=config.get_param_config())
        model_builder.build_model()
        logger.info("%s completed\nx==========x", stage_name)
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    try:
        model_builder_pipeline()
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
