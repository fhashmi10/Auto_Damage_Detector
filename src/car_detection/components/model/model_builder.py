"""Module to build base models"""
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from src.car_detection.entities.config_entity import ModelConfig, ParamConfig
from src.utils.common import create_directories
from src import logger


class ModelBuilder():
    """Class to build base models"""

    def __init__(self, config: ModelConfig, params: ParamConfig):
        self.config = config
        self.params = params

    def get_base_model(self):
        """Method to get base model"""
        try:
            base_model = hub.load(self.config.model_url)
            return base_model
        except AttributeError as ex:
            raise ex
        except Exception as ex:
            raise ex

    def update_base_model(self):
        """Method to update base model"""
        try:
            input_shape = (self.params.image_pixel_size,
                           self.params.image_pixel_size, 3)
            
            # Model architecture 
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                hub.KerasLayer(self.config.base_model_path,
                               trainable=self.params.trainable),
                tf.keras.layers.Dropout(rate=self.params.dropout_rate),
                tf.keras.layers.Dense(self.params.number_classes,
                                      kernel_regularizer=
                                      tf.keras.regularizers.l2(self.params.l2_pentaly_rate))
            ])

            # Build model
            model.build((None,)+input_shape)

            # Compile model
            sgd_optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.params.learning_rate, momentum=0.9)
            loss_crossentropy = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=0.1)
            model.compile(optimizer=sgd_optimizer,
                          loss=loss_crossentropy,
                          metrics=['accuracy'])
            return model
        except Exception as ex:
            raise ex

    @staticmethod
    def skip_processing(model_path: Path) -> bool:
        """Method to check if processing should be skipped"""
        try:
            # Delete existing model
            if os.path.exists(model_path):
                logger.info(
                    "Skip processing. Model already exists at: %s", model_path)
                return True
            # Create directory to save model
            create_directories([model_path])
            return False
        except Exception as ex:
            raise ex

    def build_model(self):
        """Method to invoke model building"""
        try:
            # Check if base model already exists
            skip_process = self.skip_processing(
                model_path=self.config.base_model_path)
            if not skip_process:
                # Get base model
                base_model = self.get_base_model()
                # Save base model
                tf.saved_model.save(base_model, self.config.base_model_path)
                logger.info("Base Model downloaded and saved successfully to: %s",
                            self.config.built_model_path)

            # Check if updated model already exists
            skip_process = self.skip_processing(
                model_path=self.config.built_model_path)
            if not skip_process:
                # Update base model
                logger.info("Updating base model.")
                model = self.update_base_model()
                # Save updated base model
                tf.saved_model.save(model, self.config.built_model_path)
                logger.info("Model built and saved successfully to: %s",
                            self.config.built_model_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
