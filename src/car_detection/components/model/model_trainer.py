"""Module to train models"""
import tensorflow as tf
from src.car_detection.configuration.configuration_manager import DataConfig,ModelConfig,ParamConfig
from src.utils.helper import build_dataset
from src import logger


class ModelTrainer():
    """Class to train models"""

    def __init__(self, data_config: DataConfig, model_config: ModelConfig, params: ParamConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.params = params

    def get_built_model(self):
        """Method to get the built model"""
        try:
            return tf.keras.models.load_model(self.model_config.built_model_path)
        except AttributeError as ex:
            logger.exception("Error loading built model.")
            raise ex
        except Exception as ex:
            raise ex

    def train_model(self):
        """Method to invoke model training"""
        try:
            # Define image size
            image_size = tuple(map(int, self.params.image_size.split(',')))
            if len(image_size)==3:
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
            transformer_model = tf.saved_model.load(self.model_config.transform_model_path)
            
            # Apply transformer model on data
            train_ds = train_ds.map(lambda images, labels:
                        (transformer_model(images), labels))
            val_ds = val_ds.map(lambda images, labels:
                                (transformer_model(images), labels))
            logger.info("Data transformed successfully.")
            
            # Get built model to train
            model = self.get_built_model()

            
            # Train the model
            steps_per_epoch = train_size // self.params.batch_size
            validation_steps = val_size // self.params.batch_size
            logger.info("Training model...")
            model.fit(train_ds,
                      epochs=self.params.number_epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=val_ds,
                      validation_steps=validation_steps,
                      verbose=2)
            #model.save(self.config.trained_model_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
