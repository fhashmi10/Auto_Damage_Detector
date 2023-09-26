"""Module to hold all common util methods"""
import os
import shutil
from pathlib import Path
from PIL import Image
import json
import base64
import pickle
import yaml
import numpy as np
import tensorflow as tf
from box import ConfigBox, exceptions

from src import logger


def read_yaml_dict(path_to_yaml: Path) -> dict:
    """Method to read yaml and return a dict instance"""
    try:
        with open(path_to_yaml, encoding='utf8') as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
            if yaml_content is None:
                raise IOError
            logger.info("yaml file loaded successfully: %s", path_to_yaml)
            return yaml_content
    except FileNotFoundError as ex:
        logger.exception("yaml file not found: %s", path_to_yaml)
        raise ex
    except IOError as ex:
        logger.exception("yaml file is empty: %s", path_to_yaml)
        raise ex
    except Exception as ex:
        raise ex


def read_yaml_configbox(path_to_yaml: Path) -> ConfigBox:
    """Method to read yaml and return a ConfigBox instance
    A configbox helps in accessing yaml contents with . syntax"""
    try:
        with open(path_to_yaml, encoding='utf8') as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
            logger.info("yaml file loaded successfully: %s", path_to_yaml)
            return ConfigBox(yaml_content)
    except exceptions.BoxValueError as ex:
        logger.exception("yaml file is empty: %s", path_to_yaml)
        raise ex
    except Exception as ex:
        raise ex


def create_directories(path_to_directories: list, is_file_path=False):
    """Method to create directories"""
    try:
        for dir_path in path_to_directories:
            if not os.path.exists(dir_path):
                if is_file_path:
                    dir_path = os.path.dirname(os.path.abspath(dir_path))
                os.makedirs(dir_path, exist_ok=True)
                logger.info("created directory at: %s", dir_path)
    except IOError as ex:
        logger.exception("Error creating directories: %s", path_to_directories)
        raise ex
    except Exception as ex:
        raise ex


def remove_directories(path_to_directories: list):
    """Method to remove directories"""
    try:
        for path in path_to_directories:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logger.info("removed directory at %s:", path)
                elif os.path.isfile(path):
                    os.remove(path)
                    logger.info("removed file at %s:", path)
    except IOError as ex:
        logger.exception("Error removing directories: %s", path_to_directories)
        raise ex
    except Exception as ex:
        raise ex


def save_object(obj, file_path: Path):
    """Method to save an object to a file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except IOError as ex:
        logger.exception("Error saving object at: %s", file_path)
        raise ex
    except Exception as ex:
        raise ex


def load_object(file_path: Path):
    """Method to load an object from a file"""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except IOError as ex:
        logger.exception("Error loading object from: %s", file_path)
        raise ex
    except Exception as ex:
        raise ex


def save_json(file_path: Path, data: dict):
    """Method to dump data to a json file"""
    try:
        create_directories([file_path], is_file_path=True)
        with open(file_path, "w", encoding="utf8") as file:
            json.dump(data, file, indent=4)
        logger.info("json file saved at: %s", file_path)
    except IOError as ex:
        logger.exception("Error loading object from: %s", file_path)
        raise ex
    except Exception as ex:
        raise ex


def load_json(file_path: Path) -> ConfigBox:
    """Method to load data from a json file"""
    try:
        with open(file_path, encoding="utf8") as file:
            content = json.load(file)
        logger.info("json file loaded succesfully from: %s", file_path)
        return ConfigBox(content)
    except IOError as ex:
        logger.exception("Error loading json from: %s", file_path)
        raise ex
    except Exception as ex:
        raise ex


def decode_image(imgstring, file_path):
    """Method to decode an image"""
    try:
        imgdata = base64.b64decode(imgstring)
        with open(file_path, 'wb') as file:
            file.write(imgdata)
            file.close()
    except IOError as ex:
        logger.exception("Error decoding image at: %s", file_path)
        raise ex
    except Exception as ex:
        raise ex

# from here
def load_file_as_list(file_path: Path):
    """Method to load data from a file as list"""
    try:
        contents_list = []
        with open(file_path, encoding="utf8") as file:
            contents = file.readlines()
            contents_list = [content.strip() for content in contents]

        logger.info("File loaded succesfully from: %s", file_path)
        return contents_list
    except IOError as ex:
        logger.exception("Error loading json from: %s", file_path)
        raise ex
    except Exception as ex:
        raise ex


def preprocess_image(image_dict, image_size=256, dynamic_size=False, max_dynamic_size=512):
    """Method to preprocess an image"""
    # Open image and convert to float nd array (float32 is default dtype in img_to_array)
    # Floating point value images are expected to have values in the range [0,1]
    # Integer data type images are expected to have values in the range [0,MAX]
    # where MAX is the largest positive representable number for the data type.
    image = tf.keras.preprocessing.image.img_to_array(Image.open(image_dict))
    # Normalize to range [0, 1].
    if tf.reduce_max(image) > 1.0:
        image = image / 255.
    # Reshape into [batch_size, height, width, num_channels]
    image = np.expand_dims(image, axis=0)
    # Stacking if black and white image to have same shape
    if len(image.shape) == 3:
        image = tf.stack([image, image, image], axis=-1)
    # Resize image
    if not dynamic_size:
        image = tf.image.resize_with_pad(image, image_size, image_size)
    # Resize - some models use a dynamic input size (enabling inference on the unscaled image)
    elif image.shape[1] > max_dynamic_size or image.shape[2] > max_dynamic_size:
        image = tf.image.resize_with_pad(
            image, max_dynamic_size, max_dynamic_size)
    return image
