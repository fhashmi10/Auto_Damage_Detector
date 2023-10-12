"""Module to create training pipeline"""
from src import logger
from src.image_classification.configuration.configuration_manager import ConfigurationManager
from src.image_classification.components.data.data_ingestion import DataIngestion
from src.image_classification.components.model.model_builder import ModelBuilder
from src.image_classification.components.model.model_callbacks import ModelCallbacks
from src.image_classification.components.model.model_trainer import ModelTrainer
from src.image_classification.components.model.model_evaluator import ModelEvaluator


class ImageClassificationTrainingPipeline:
    """Class to create training pipeline"""

    def __init__(self):
        pass

    def data_ingestion(self, config: ConfigurationManager, stage_name: str):
        """Method to perform data ingestion"""
        try:
            logger.info("%s started", stage_name)
            data_ingestion = DataIngestion(config=config.get_data_config())
            data_ingestion.ingest_data()
            logger.info("%s completed\nx==========x", stage_name)
        except Exception as ex:
            raise ex

    def model_builder(self, config: ConfigurationManager, stage_name: str):
        """Method to perform model building"""
        try:
            logger.info("%s started", stage_name)
            model_builder = ModelBuilder(config=config.get_model_config(),
                                         params=config.get_param_config())
            model_builder.build_model()
            logger.info("%s completed\nx==========x", stage_name)
        except Exception as ex:
            raise ex

    def model_trainer(self, config: ConfigurationManager, stage_name: str):
        """Method to perform model training"""
        try:
            logger.info("%s started", stage_name)
            model_callback = ModelCallbacks(
                config=config.get_callback_config())
            callback_list = model_callback.get_callbacks()

            model_trainer = ModelTrainer(data_config=config.get_data_config(),
                                         model_config=config.get_model_config(),
                                         callback_config=config.get_callback_config(),
                                         params=config.get_param_config())
            model_trainer.train_model(callback_list=callback_list)
            logger.info("%s completed\nx==========x", stage_name)
        except Exception as ex:
            raise ex

    def model_evaluator(self, config: ConfigurationManager, stage_name: str):
        """Method to perform model evaluation"""
        try:
            logger.info("%s started", stage_name)
            model_evaluator = ModelEvaluator(data_config=config.get_data_config(),
                                             model_config=config.get_model_config(),
                                             params=config.get_param_config(),
                                             eval_config=config.get_evaluation_config())
            model_evaluator.evaluate_model()
            logger.info("%s completed\nx==========x", stage_name)
        except Exception as ex:
            raise ex

    def run_pipeline(self, config_file_path, params_file_path, steps: int, stage: str):
        """Method to perform training"""
        try:
            if steps!=0:
                config = ConfigurationManager(config_file_path=config_file_path,
                                            params_file_path=params_file_path)
                if steps>=1:
                    self.data_ingestion(
                        config=config, stage_name=stage+": Data Ingestion")
                if steps>=2:
                    self.model_builder(
                        config=config, stage_name=stage+": Model Building")
                if steps>=3:
                    self.model_trainer(
                        config=config, stage_name=stage+": Model Training")
                if steps>=4:
                    self.model_evaluator(
                        config=config, stage_name=stage+": Model Evaluation")
            else:
                logger.info("Please provide number of steps to run min 1 to max 4.")
        except Exception as ex:
            raise ex
