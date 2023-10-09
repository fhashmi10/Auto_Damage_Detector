"""Module to create model trainer pipeline"""
from src.damage_detection.configuration.damage_detection_configuration_manager \
    import DamageDetectionConfigurationManager
from src.damage_detection.components.model.model_callbacks import ModelCallbacks
from src.damage_detection.components.model.model_trainer import ModelTrainer
from src import logger

def model_trainer_pipeline():
    """Method to perform model training"""
    try:
        stage_name = "Car Detection Model Training"
        logger.info("%s started", stage_name)
        config = DamageDetectionConfigurationManager()
        model_callback = ModelCallbacks(config=config.get_callback_config())
        callback_list = model_callback.get_callbacks()
        model_trainer = ModelTrainer(data_config=config.get_data_config(),
                                     model_config=config.get_model_config(),
                                     callback_config=config.get_callback_config(),
                                     params=config.get_param_config())
        model_trainer.train_model(callback_list=callback_list)
        logger.info("%s completed\nx==========x", stage_name)
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    try:
        model_trainer_pipeline()
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
