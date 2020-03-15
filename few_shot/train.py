import time
import os
import json
from typing import Dict
from meta_learning.datasets import create_dataset


def train(config_file: Dict):
    print('Initialise dataset...')
    train_dataset = create_dataset(config_file['train_dataset_params'])
    train_dataset_size = len(train_dataset)
    print(f'The number of training sample = {train_dataset_size}')

    val_dataset = create_dataset(config_file['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print(f'The number of validation samples = {val_dataset_size}')

    print('Initialise model...')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    configuration_file = json.load('config.json')

    # SageMaker env parameters
    if os.environ.get('SM_CHANNEL_TRAIN', None):
        output_dir = os.environ['SM_OUTPUT_DATA_DIR']
        configuration_file[
            'train_dataset_params'
        ]['dataset_path'] = os.environ['SM_CHANNEL_TRAIN']
        configuration_file[
            'val_dataset_params'
        ]['dataset_path'] = os.environ['SM_CHANNEL_TEST']

    # TODO: load params via SageMaker if hyperparameter search

    train(configuration_file)
