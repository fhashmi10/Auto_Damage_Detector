"""Module to create training pipeline"""
import sys
from src import logger
from src.damage_severity import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.image_classification.pipeline.\
    image_classification_training_pipeline import ImageClassificationTrainingPipeline


def run_ds_pipeline(steps: int):
    """Method to perform training"""
    try:
        pipeline = ImageClassificationTrainingPipeline()
        pipeline.run_pipeline(config_file_path=CONFIG_FILE_PATH,
                                params_file_path=PARAMS_FILE_PATH,
                                steps=steps,
                                stage="Damage Severity")
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    try:
        if len(sys.argv)>1:
            NUM_STEPS = int(sys.argv[1])
        else:
            NUM_STEPS = 0
        run_ds_pipeline(steps=NUM_STEPS)
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
