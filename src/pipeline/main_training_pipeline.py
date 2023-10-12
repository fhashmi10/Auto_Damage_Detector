"""Module to create Main training pipeline"""
import sys
from src.car_detection.pipeline.car_detection_training_pipeline import run_cd_pipeline
from src.damage_detection.pipeline.damage_detection_training_pipeline import run_dd_pipeline
from src.damage_severity.pipeline.damage_severity_training_pipeline import run_ds_pipeline
from src import logger


class MainTrainingPipeline:
    """Class to create Main training pipeline"""

    def __init__(self):
        pass

    def run_pipeline(self, steps: int):
        """Method to perform main training"""
        try:
            # Car detection
            run_cd_pipeline(steps=0)
            # Damage detection
            run_dd_pipeline(steps=0)
            # Damage severity
            run_ds_pipeline(steps=steps)
        except Exception as ex:
            raise ex


if __name__ == '__main__':
    try:
        if len(sys.argv)>1:
            NUM_STEPS = int(sys.argv[1])
        else:
            NUM_STEPS = 0
        obj = MainTrainingPipeline()
        obj.run_pipeline(steps=NUM_STEPS)
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
