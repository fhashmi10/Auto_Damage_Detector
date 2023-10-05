"""Module to evaluate models"""
from urllib.parse import urlparse
import mlflow
import tensorflow as tf
from src.car_detection.entities.config_entity import \
    ModelConfig, ParamConfig, EvaluationConfig
from src.utils.helper import build_dataset
from src.utils.common import save_json
from src import logger


class ModelEvaluator():
    """Class to evaluate models"""

    def __init__(self, model_config: ModelConfig,
                 params: ParamConfig, eval_config: EvaluationConfig):
        self.model_config = model_config
        self.params = params
        self.eval_config = eval_config

    def get_trained_model(self):
        """Method to get the trained model"""
        try:
            return tf.keras.models.load_model(self.model_config.trained_model_path)
        except AttributeError as ex:
            logger.exception("Error loading trained model.")
            raise ex
        except Exception as ex:
            raise ex

    def get_dataset(self):
        """Method to get dataset"""
        try:
            # Define image size
            image_size = tuple(map(int, self.params.image_size.split(',')))
            if len(image_size) == 3:
                image_size = image_size[:2]

            # Build test dataset
            test_ds, _ = build_dataset(data_dir=self.eval_config.test_data_path,
                                                 val_split=None,
                                                 subset=None,
                                                 image_size=image_size,
                                                 batch_size=self.params.batch_size)
            logger.info("Data loaded successfully.")

            return test_ds
        except AttributeError as ex:
            logger.exception("Error getting dataset.")
            raise ex
        except Exception as ex:
            raise ex

    def log_mlflow(self, model, model_score: dict):
        """Method to log to MLflow"""
        try:
            # Below URL can be used to save experiments on remote server (dagshub can be used)
            # dagshub uri, username and password will need to be
            # exported as env variabls using gitbash terminal
            mlflow.set_registry_uri(self.eval_config.mlflow_uri)
            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(self.eval_config.track_params)
                mlflow.log_metric("loss", model_score["loss"])
                mlflow.log_metric("accuracy", model_score["accuracy"])

                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(
                        model, "model", registered_model_name="abc")
                else:
                    mlflow.sklearn.log_model(model, "model")
        except AttributeError as ex:
            raise ex
        except Exception as ex:
            raise ex

    def evaluate_model(self):
        """Method to invoke model training"""
        # Get base model
        try:
            # Get trained model
            model = self.get_trained_model()

            # Get test data
            test_ds = self.get_dataset()

            # Evaluate
            model_score = model.evaluate(test_ds)
            result = {"loss": model_score[0], "accuracy": model_score[1]}
            save_json(
                file_path=self.eval_config.evaluation_score_json_path, data=result)

            # Log ml flow
            self.log_mlflow(model, result)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
