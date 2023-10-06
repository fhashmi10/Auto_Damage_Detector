"""Module to create Main training pipeline"""
from src.car_detection.pipeline.training_pipeline.car_detection_training_pipeline \
    import CarDetectionTrainingPipeline
from src import logger


class MainTrainingPipeline:
    """Class to create Main training pipeline"""

    def __init__(self):
        pass

    def run_pipeline(self):
        """Method to perform main training"""
        try:
            car_detection = CarDetectionTrainingPipeline()
            car_detection.run_pipeline()
        except Exception as ex:
            raise ex


if __name__ == '__main__':
    try:
        obj = MainTrainingPipeline()
        obj.run_pipeline()
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
