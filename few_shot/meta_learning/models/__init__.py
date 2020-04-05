import importlib
import logging
from typing import Dict
from meta_learning.models.base_model import BaseModel


def find_model_using_name(model_name: str):
    model_filename = f'meta_learning.models.{model_name}_model'
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
         and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        logging.warning(
            f'''In {model_filename}.py, there should be a subclass of
            BaseModel with class name that matches {target_model_name}
            in lowercase'''
        )
        exit(0)

    return model


def create_model(configuration: Dict):
    """Create a model given the configuration.

    Arguments:
        configuration {Dict} -- The model configuration.

    Returns:
        A model object.
    """
    model = find_model_using_name(configuration['model_name'])
    model_object = model(configuration)
    logging.info(f'model [{type(model_object).__name__}] was created')
    return model_object
