
"""Module to perform prediction"""
import tensorflow as tf
from flask import current_app
from src.image_classification.entities.config_entity import DataConfig, ModelConfig, ParamConfig
from src.utils.common import preprocess_image, load_file_as_list
from src import logger


class ModelPredictor:
    """Class to perform prediction"""

    def __init__(self, data_config: DataConfig, model_config: ModelConfig, params: ParamConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.params = params

    def get_class_labels(self) -> list:
        """Return the class labels"""
        try:
            classes = load_file_as_list(self.data_config.class_labels_path)
            return classes
        except AttributeError as ex:
            raise ex
        except Exception as ex:
            raise ex

    def predict(self, image_dict):
        """Method to invoke prediction"""
        try:
            # Transform input data
            input_image = preprocess_image(
                image_dict=image_dict, image_size=int(self.params.image_size.split(',')[0]))

            # Load the model
            # image_classification_model = tf.keras.models.load_model(
            #     self.model_config.trained_model_path)
            image_classification_model = current_app.image_classification_model
            logger.info("loaded model successfully.")

            # Get class labels
            classes = self.get_class_labels()

            # Predict
            predict_proba = tf.nn.softmax(
                image_classification_model(input_image)).numpy()
            top_prediction = tf.argsort(
                predict_proba, axis=-1, direction="DESCENDING")[0][:1].numpy()[0]
            logger.info("Prediction completed.")

            # Return Prediction
            prediction = classes[top_prediction]
            prediction = []
            if classes[top_prediction] == "undamaged":
                prediction.append(False)
            else:
                prediction.append(True)
            prediction.append(classes[top_prediction].replace("_"," "))

            return prediction
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
