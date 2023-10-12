"""Module to perform callbacks"""
#import os
#import time
import tensorflow as tf
from src.image_classification.entities.config_entity import CallbackConfig
from src import logger


class ModelCallbacks():
    """Class to perform callbacks"""

    def __init__(self, config: CallbackConfig):
        self.config = config

    # @property
    # def _create_tb_callbacks(self):
    #     """Property to hold tensorboard log details"""
    #     try:
    #         timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    #         tb_running_log_dir = os.path.join(
    #             self.config.tensorboard_log_path,
    #             f"tb_logs_at_{timestamp}",
    #         )
    #         return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    #     except AttributeError as ex:
    #         raise ex

    @property
    def _create_ckpt_callbacks(self):
        """Property to hold model checkpoints"""
        try:
            return tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.model_checkpoint_path,
                monitor="accuracy",
                # accuracy since val loss/accuracy is only calculated after epoch
                save_best_only=True,
                save_freq=50#number of batches - by default save after each epoch
            )
        except AttributeError as ex:
            raise ex

    @property
    def _create_lr_callbacks(self):
        """Property to hold learning rate"""
        try:
            return tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, min_lr=0.001)
        except AttributeError as ex:
            raise ex

    @property
    def _create_earlystop_callbacks(self):
        """Property to define early stopping"""
        try:
            return tf.keras.callbacks.EarlyStopping(patience=3)#keeping patience 1 more than lr
        except AttributeError as ex:
            raise ex

    def get_callbacks(self):
        """Method to invoke callbacks"""
        try:
            return [self._create_ckpt_callbacks,
                self._create_lr_callbacks,
                self._create_earlystop_callbacks]
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
