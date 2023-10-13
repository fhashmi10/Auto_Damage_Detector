"""Module to create model prediction pipeline"""
from src import logger, MODEL_KEY_DD
from src.damage_detection import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.image_classification.pipeline.\
    image_classification_prediction_pipeline import image_classification_prediction


def predict_dd(filename) -> list:
    """Method to perform prediction"""
    try:
        prediction = image_classification_prediction(model_key=MODEL_KEY_DD,
                                                     config_file_path=CONFIG_FILE_PATH,
                                                     params_file_path=PARAMS_FILE_PATH,
                                                     filename=filename)
        pred_result = "The damage area is: " + prediction
        prediction_list = []
        if prediction == "undamaged":
            prediction_list.append(False)
            pred_result = "No damage identified."
        elif prediction == "totaled":
            prediction_list.append(False)
            pred_result = "The vehicle is totaled."
        else:
            prediction_list.append(True)
        prediction_list.append(pred_result)
        return prediction_list
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
