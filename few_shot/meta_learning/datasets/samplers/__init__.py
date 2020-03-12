import importlib
from meta_learning.datasets.samplers.base_sampler import BaseSampler
from typing import Dict

samplers = ['episodic']  # list of custom samplers


def check_for_sampler(dataset_name: str) -> bool:
    return dataset_name.split('_')[0] in samplers


def find_sampler_using_name(dataset_name: str):
    sampler_filename = f'''meta_learning.datasets.samplers.{dataset_name.split('_')[0]}_sampler'''
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
    sampler = find_sampler_using_name(configuration['dataset_name'])
    return sampler(dataset, configuration['sampler_params'])
