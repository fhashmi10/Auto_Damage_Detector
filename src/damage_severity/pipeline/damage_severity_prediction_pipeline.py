"""Module to create model prediction pipeline"""
from src import logger, MODEL_KEY_DS
from src.damage_severity import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.image_classification.pipeline.\
    image_classification_prediction_pipeline import image_classification_prediction


def predict_ds(filename) -> str:
    """Method to perform prediction"""
    try:
        prediction = image_classification_prediction(model_key=MODEL_KEY_DS,
                                                     config_file_path=CONFIG_FILE_PATH,
                                                     params_file_path=PARAMS_FILE_PATH,
                                                     filename=filename)
        pred_result = "The damage severity is: " + prediction
        return pred_result
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
