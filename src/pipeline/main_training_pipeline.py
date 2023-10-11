"""Module to create Main training pipeline"""
from src.car_detection.pipeline.training_pipeline.car_detection_training_pipeline \
    import CarDetectionTrainingPipeline
from src.damage_detection.pipeline.training_pipeline.damage_detection_training_pipeline \
    import DamageDetectionTrainingPipeline
from src.damage_severity.pipeline.training_pipeline.damage_severity_training_pipeline \
    import DamageSeverityTrainingPipeline
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
            damage_detection = DamageDetectionTrainingPipeline()
            damage_detection.run_pipeline()
            damage_severity = DamageSeverityTrainingPipeline()
            damage_severity.run_pipeline()
        except Exception as ex:
            raise ex


if __name__ == '__main__':
    try:
        obj = MainTrainingPipeline()
        obj.run_pipeline()
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
