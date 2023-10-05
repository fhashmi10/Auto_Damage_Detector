"""Module to define frequently used functions that are not too generic to fall under common.py"""
import tensorflow as tf
from src import logger


def build_dataset(data_dir, val_split, subset, image_size, batch_size):
    """Method to build dataset"""
    try:
        image_data = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=val_split,
            subset=subset,
            label_mode="categorical",
            # Seed needs to provided when using validation_split and shuffle=True (default).
            # A fixed seed is used so that the validation set is stable across runs.
            seed=123,
            image_size=image_size,
            batch_size=1)
        # Get data size
        data_size = image_data.cardinality().numpy()
        # Batch data as now we got data size
        image_data = image_data.unbatch().batch(batch_size)
        # Repeat to create infinite dataset - steps must be defined then
        # image_data = image_data.repeat()

        return image_data, data_size
    except OSError as ex:
        logger.exception("Error reading data")
        raise ex
    except Exception as ex:
        raise ex
