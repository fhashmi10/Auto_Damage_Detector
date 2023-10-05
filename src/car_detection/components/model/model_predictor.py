
"""Module to perform prediction"""
import tensorflow as tf
import tensorflow_hub as hub
from src.car_detection.entities.config_entity import ModelConfig
from src.utils.common import preprocess_image, load_file_as_list
from src import logger


class ModelPredictor:
    """Class to perform prediction"""

    def __init__(self, config=ModelConfig):
        self.config = config

    def get_class_labels(self) -> list:
        """Return the class labels"""
        try:
            classes = load_file_as_list(self.config.labels_file_path)
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
            car_detection_model = hub.load(self.config.trained_model_path)
            logger.info("loaded model successfully.")

            # Get class labels
            #classes = self.get_class_labels()
            classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
            # Predict
            prediction = "Image is not identified as a car."
            predict_proba = tf.nn.softmax(
                car_detection_model(input_image)).numpy()
            top_prediction = tf.argsort(
                predict_proba, axis=-1, direction="DESCENDING")[0][:1].numpy()[0]
            
            # top_five_predictions = tf.argsort(
            #     predict_proba, axis=-1, direction="DESCENDING")[0][:5].numpy()

            # for item in top_five_predictions:
            #     #class_index = item + 1
            #     class_index = item
            #     if "automobile" in classes[class_index]:
            #         prediction = "Image is identified as a car."
            if "automobile" in classes[top_prediction]:
                prediction = "Image is identified as a car."
            logger.info("Prediction completed.")

            return prediction
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
