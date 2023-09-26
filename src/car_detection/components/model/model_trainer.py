"""Module to train models"""
import tensorflow as tf
from src.car_detection.configuration.configuration_manager import DataConfig,TrainConfig
from src import logger


class ModelTrainer():
    """Class to train models"""

    def __init__(self, data_config: DataConfig, model_config: TrainConfig):
        self.data_config = data_config
        self.model_config = model_config

    def get_base_model(self):
        """Method to get the base model"""
        try:
            return tf.keras.models.load_model(self.config.base_model_path)
        except AttributeError as ex:
            logger.exception("Error loading base model.")
            raise ex
        except Exception as ex:
            raise ex

    def train_model(self, callback_list: list):
        """Method to invoke model training"""
        try:
            # Get base model
            model = self.get_base_model()

            # Get train and valid generator
            image_data_generator = DataGenerator.instance()
            image_data_generator.create_image_data_generator(self.config)
            train_generator = image_data_generator.get_train_generator()
            valid_generator = image_data_generator.get_valid_generator()

            steps_per_epoch = train_generator.samples // train_generator.batch_size
            validation_steps = valid_generator.samples // valid_generator.batch_size

            # Train model
            model.fit(train_generator,
                        epochs=self.config.params_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        validation_data=valid_generator,
                        callbacks=callback_list
                        )
            model.save(self.config.trained_model_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
