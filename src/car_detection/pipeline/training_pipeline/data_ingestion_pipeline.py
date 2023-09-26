"""Module to create data ingestion pipeline"""
from src.car_detection.configuration.configuration_manager import ConfigurationManager
from src.car_detection.components.data.data_ingestion import DataIngestion
from src import logger

def data_ingestion_pipeline():
    """Method to perform data ingestion"""
    try:
        stage_name = "Car Detection Data Ingestion"
        logger.info("%s started", stage_name)
        config = ConfigurationManager()
        data_ingestion = DataIngestion(config=config.get_data_config())
        data_ingestion.ingest_data()
        logger.info("%s completed\nx==========x", stage_name)
    except Exception as ex:
        raise ex


if __name__ == '__main__':
    try:
        data_ingestion_pipeline()
    except Exception as exc:
        logger.exception("Exception occured: %s", exc)
