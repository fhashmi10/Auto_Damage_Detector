""""Module to perform data ingestion"""
import os
import tensorflow as tf
from src.image_classification.entities.config_entity import DataConfig
from src.utils.common import create_directories
from src import logger


class DataIngestion():
    """Class to perform data ingestion"""

    def __init__(self, config: DataConfig):
        self.config = config

    def ingest_data(self):
        """Method to ingest data"""
        try:
            if self.config.source_url is not None:
                create_directories([self.config.data_path])
                data_save_abs_path = os.path.join("./", self.config.data_path)
                tf.keras.utils.get_file(
                    origin=self.config.source_url,
                    extract=True,
                    cache_dir=data_save_abs_path)
                logger.info("Data downloaded at: %s", self.config.data_path)
            else:
                logger.info("Data will be read from: %s", self.config.data_path)
        except AttributeError as ex:
            logger.exception("Error finding attribute: %s", ex)
            raise ex
        except Exception as ex:
            logger.exception("Exception occured: %s", ex)
            raise ex
