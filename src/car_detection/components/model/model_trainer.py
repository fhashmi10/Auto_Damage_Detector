"""Module to train models"""
import os
from pathlib import Path
import tensorflow as tf
from src.car_detection.configuration.configuration_manager import \
    DataConfig, ModelConfig, CallbackConfig, ParamConfig
from src.utils.helper import build_dataset
from src import logger


class ModelTrainer():
    """Class to train models"""

    def __init__(self, data_config: DataConfig, model_config: ModelConfig, callback_config: CallbackConfig, params: ParamConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.callback_config = callback_config
        self.params = params

    @staticmethod
    def model_checkpoint_exists(checkpoint_path: Path) -> bool:
        """Method to check if model checkpoint exists"""
        try:
            # Delete existing model
            if os.path.exists(checkpoint_path):
                logger.info(
                    "Model checkpoint already exists at: %s", checkpoint_path)
                return True
            return False
        except Exception as ex:
            raise ex

    @staticmethod
    def get_model(model_path: Path):
        """Method to get the model"""
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except AttributeError as ex:
            logger.exception("Error loading built model.")
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

            # Build training dataset
            train_ds, train_size = build_dataset(data_dir=self.data_config.data_path,
                                                 val_split=0.2,
                                                 subset="training",
                                                 image_size=image_size,
                                                 batch_size=self.params.batch_size)

            # Build validation dataset
            val_ds, val_size = build_dataset(data_dir=self.data_config.data_path,
                                             val_split=0.2,
                                             subset="validation",
                                             image_size=image_size,
                                             batch_size=self.params.batch_size)
            logger.info("Data loaded successfully.")

            # Load transformer model
            transformer_model = tf.saved_model.load(
                self.model_config.transform_model_path)

            # Apply transformer model on data
            train_ds = train_ds.map(lambda images, labels:
                                    (transformer_model(images), labels))
            val_ds = val_ds.map(lambda images, labels:
                                (transformer_model(images), labels))
            logger.info("Data transformed successfully.")

            return train_ds, train_size, val_ds, val_size
        except AttributeError as ex:
            logger.exception("Error getting dataset.")
            raise ex
        except Exception as ex:
            raise ex

    def train_model(self, callback_list: list):
        """Method to invoke model training"""
        try:

            # Get train and validate data
            train_ds, train_size, val_ds, val_size = self.get_dataset()

            # Get built model to train
            checkpoint_path = os.path.join(self.callback_config.callback_path, "checkpoints")
            if self.model_checkpoint_exists(checkpoint_path):
                model = self.get_model(checkpoint_path)
            else:
                model = self.get_model(self.model_config.built_model_path)

            # Train the model
            steps_per_epoch = train_size // self.params.batch_size
            validation_steps = val_size // self.params.batch_size
            logger.info("Training model...")
            model.fit(train_ds,
                      epochs=self.params.number_epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=val_ds,
                      validation_steps=validation_steps,
                      callbacks=callback_list)

            # Save trained model
            model.save(self.model_config.trained_model_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
