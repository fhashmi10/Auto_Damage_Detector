
"""Module to perform prediction"""
import tensorflow as tf
from src.car_detection.entities.config_entity import DataConfig, ModelConfig
from src.utils.common import preprocess_image, load_file_as_list
from src import logger


class ModelPredictor:
    """Class to perform prediction"""

    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        self.data_config = data_config
        self.model_config = model_config

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
                image_dict=image_dict, image_size=384)

            # Load the model
            car_detection_model = tf.keras.models.load_model(
                self.model_config.trained_model_path)
            logger.info("loaded model successfully.")

            # Get class labels
            classes = self.get_class_labels()

            # Predict
            predict_proba = tf.nn.softmax(
                car_detection_model(input_image)).numpy()
            top_prediction = tf.argsort(
                predict_proba, axis=-1, direction="DESCENDING")[0][:1].numpy()[0]
            logger.info("Prediction completed.")

            # Return Prediction
            prediction = []
            if classes[top_prediction] == "automobile":
                prediction.append(True)
            else:
                prediction.append(False)
            prediction.append(classes[top_prediction])

            return prediction
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
