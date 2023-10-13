"""Module to create model prediction pipeline"""
from src import logger, MODEL_KEY_CD
from src.car_detection import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.image_classification.pipeline.\
    image_classification_prediction_pipeline import image_classification_prediction


def predict_cd(filename) -> list:
    """Method to perform prediction"""
    try:
        prediction = image_classification_prediction(model_key=MODEL_KEY_CD,
                                                     config_file_path=CONFIG_FILE_PATH,
                                                     params_file_path=PARAMS_FILE_PATH,
                                                     filename=filename)
        pred_result = "The image is identified as: " + prediction
        prediction_list = []
        if prediction == "automobile":
            prediction_list.append(True)
        else:
            pred_result += ". Please upload car damage image."
            prediction_list.append(False)
        prediction_list.append(pred_result)
        return prediction_list
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
