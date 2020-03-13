import importlib
from meta_learning.datasets.samplers.base_sampler import BaseSampler
from typing import Dict

samplers = ['episodic']  # list of custom samplers


def check_for_sampler(dataset_name: str) -> bool:
    """Checks if there is a dedicated custom sampler for a dataset.
    The custom samplers should have the following name convention:
    [dataset_name]_sampler.py with the class inside the script having
    name [Dataset_name]Sampler().

    Arguments:
        dataset_name {str} -- Name of the dataset in the configuration file.

    Returns:
        bool -- Whether there is a custom sampler available.
    """
    return dataset_name.split('_')[0] in samplers


def find_sampler_using_name(dataset_name: str):
    """Import module meta_learning/datasets/samplers/[dataset_name]_sampler.py

    Arguments:
        dataset_name {str} -- Prefix of module name and name for class
            instantiation, which is DatasetNameSampler().

    Raises:
        NotImplementedError: whether DatasetNameSampler() is implemented in
            [dataset_name]_sampler.py.

    Returns:
        cls -- Class DatasetNameSampler() from [dataset_name]_sampler.py
    """
    sampler_filename = (f"meta_learning.datasets.samplers."
                        f"{dataset_name.split('_')[0]}_sampler")
    samplerlib = importlib.import_module(sampler_filename)

    sampler = None
    target_sampler_name = dataset_name.split('_')[0] + 'sampler'
    for name, cls in samplerlib.__dict__.items():
        if name.lower() == target_sampler_name.lower():
            sampler = cls

    if sampler is None:
        raise NotImplementedError(
            f'''In {dataset_filename}, there should be a subclass of
            BaseSampler with class that matches {target_sampler_name} in
            lowercase'''
        )

    return sampler


def create_sampler(configuration: Dict, dataset):
    """Creates an sampler object given the configuration and dataset.

    Arguments:
        configuration {Dict} -- Configuration file that should include
            a `sampler_params` sub-dictionary for the sampler configuration.
        dataset {[type]} -- A dataset object.

    Returns:
        An instantiated sampler object.
    """
    sampler = find_sampler_using_name(configuration['dataset_name'])
    return sampler(dataset, configuration['sampler_params'])
