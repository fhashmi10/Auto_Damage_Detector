"""Module to create model evaluator pipeline"""
from src.damage_detection.configuration.damage_detection_configuration_manager \
    import DamageDetectionConfigurationManager
from src.damage_detection.components.model.model_evaluator import ModelEvaluator
from src import logger

def model_evaluator_pipeline():
    """Method to perform model evaluation"""
    try:
        stage_name = "Car Detection Model Evaluation"
        logger.info("%s started", stage_name)
        config = DamageDetectionConfigurationManager()
        model_evaluator = ModelEvaluator(data_config=config.get_data_config(),
                                         model_config=config.get_model_config(),
                                         params=config.get_param_config(),
                                         eval_config=config.get_evaluation_config())
        model_evaluator.evaluate_model()
        logger.info("%s completed\nx==========x", stage_name)
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    try:
        model_evaluator_pipeline()
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
