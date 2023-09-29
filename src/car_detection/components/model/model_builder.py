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
        self.input_shape = tuple(map(int, self.params.image_size.split(',')))

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
            # Model architecture
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=self.input_shape),
                hub.KerasLayer(self.config.base_model_path,
                               trainable=self.params.trainable),
                tf.keras.layers.Dropout(rate=self.params.dropout_rate),
                tf.keras.layers.Dense(self.params.number_classes,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.params.l2_pentaly_rate))
            ])

            # Build model
            model.build((None,)+self.input_shape)

            # Compile model
            sgd_optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.params.learning_rate, momentum=0.9)
            loss_crossentropy = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=0.1)
            #Label smoothing is a regularization technique for overfitting and overconfidence
            # An overconfident model predicted probabilities is higher than accuracy
            # e.g, may predict 0.9 for inputs where the accuracy is only 0.6
            model.compile(optimizer=sgd_optimizer,
                          loss=loss_crossentropy,
                          metrics=['accuracy'])

            return model
        except Exception as ex:
            raise ex

    def create_transformer_model(self) -> tf.keras.Sequential:
        """Method to create data transformer object"""
        try:
            # Rescale
            transformer_model = tf.keras.Sequential(
                [tf.keras.layers.Rescaling(1. / 255)])

            # Data augmentation
            if self.params.augmentation:
                transformer_model.add(
                    tf.keras.layers.RandomRotation(40))
                transformer_model.add(
                    tf.keras.layers.RandomTranslation(0, 0.2))
                transformer_model.add(
                    tf.keras.layers.RandomTranslation(0.2, 0))
                transformer_model.add(
                    tf.keras.layers.RandomZoom(0.2, 0.2))
                transformer_model.add(
                    tf.keras.layers.RandomFlip(mode="horizontal"))

            # Build model
            transformer_model.build((None,)+self.input_shape)

            return transformer_model
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
                # Save base model - it is important to save it as avoids re-downloads
                tf.saved_model.save(base_model,
                                    self.config.base_model_path)
                logger.info("Base Model downloaded and saved successfully to: %s",
                            self.config.built_model_path)

            # Check if updated model already exists
            skip_process = self.skip_processing(
                model_path=self.config.built_model_path)
            if not skip_process:
                # Update base model
                updated_model = self.update_base_model()
                # Save updated base model
                updated_model.save(self.config.built_model_path)
                logger.info("Updated model built and saved successfully to: %s",
                            self.config.built_model_path)

            # Check if transformer model already exists
            skip_process = self.skip_processing(
                model_path=self.config.transform_model_path)
            if not skip_process:
                # Create transformation model
                transformer_model = self.create_transformer_model()
                # Save transformation model
                tf.saved_model.save(transformer_model,
                                    self.config.transform_model_path)
                logger.info("Transformer model built and saved successfully to: %s",
                            self.config.transform_model_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
