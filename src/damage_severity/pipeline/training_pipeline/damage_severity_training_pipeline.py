"""Module to create damage detection training pipeline"""
from src import logger
from .data_ingestion_pipeline import data_ingestion_pipeline
from .model_builder_pipeline import model_builder_pipeline
from .model_trainer_pipeline import model_trainer_pipeline
from .model_evaluator_pipeline import model_evaluator_pipeline


class DamageSeverityTrainingPipeline:
    """Class to create damage detection training pipeline"""

    def __init__(self):
        pass

    def damage_severity_data_ingestion(self):
        """Method to perform data ingestion"""
        try:
            data_ingestion_pipeline()
        except Exception as ex:
            raise ex

    def damage_severity_model_builder(self):
        """Method to perform model building"""
        try:
            model_builder_pipeline()
        except Exception as ex:
            raise ex

    def damage_severity_model_trainer(self):
        """Method to perform model training"""
        try:
            model_trainer_pipeline()
        except Exception as ex:
            raise ex

    def damage_severity_model_evaluator(self):
        """Method to perform model evaluation"""
        try:
            model_evaluator_pipeline()
        except Exception as ex:
            raise ex

    def run_pipeline(self):
        """Method to perform car detection training"""
        try:
            self.damage_severity_data_ingestion()
            self.damage_severity_model_builder()
            self.damage_severity_model_trainer()
            self.damage_severity_model_evaluator()
        except Exception as ex:
            raise ex


if __name__ == '__main__':
    try:
        obj = DamageSeverityTrainingPipeline()
        obj.run_pipeline()
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
