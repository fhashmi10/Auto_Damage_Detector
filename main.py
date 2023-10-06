"""
main module
"""
from src import logger
from src.pipeline.main_training_pipeline import MainTrainingPipeline

try:
    obj = MainTrainingPipeline()
    obj.run_pipeline()
except Exception as ex:
    logger.exception("Exception in processing: %s", ex)
