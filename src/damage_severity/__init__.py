"""Define config file paths"""
from pathlib import Path
from src import logger
from src.image_classification.configuration.configuration_manager import ConfigurationManager

CONFIG_FILE_PATH = Path("config/damage_severity_config.yaml")
PARAMS_FILE_PATH = Path("config/damage_severity_params.yaml")

def get_ds_config():
    """Method to load trained model"""
    try:
        config = ConfigurationManager(config_file_path=CONFIG_FILE_PATH,
                                      params_file_path=PARAMS_FILE_PATH)
        return config
    except Exception as ex:
        logger.exception("Exception occured: %s", ex)
        raise ex
