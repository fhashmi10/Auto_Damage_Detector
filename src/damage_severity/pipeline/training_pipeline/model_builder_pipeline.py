"""Module to create model builder pipeline"""
from src.damage_severity.configuration.damage_severity_configuration_manager \
    import DamageSeverityConfigurationManager
from src.damage_severity.components.model.model_builder import ModelBuilder
from src import logger

def model_builder_pipeline():
    """Method to perform model building"""
    try:
        stage_name = "Car Detection Model Build"
        logger.info("%s started", stage_name)
        config = DamageSeverityConfigurationManager()
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