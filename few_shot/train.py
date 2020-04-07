import time
import os
import json
import duallog
from typing import Dict
from meta_learning.datasets import create_dataset
from meta_learning.models import create_model


def train(config_file: Dict):
    logger.info('Initialise dataset...')
    train_dataset = create_dataset(config_file['train_dataset_params'])
    train_dataset_size = len(train_dataset)
    logger.info(f'The number of training sample = {train_dataset_size}')

    val_dataset = create_dataset(config_file['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    logger.info(f'The number of validation samples = {val_dataset_size}')

    logger.info('Initialise model...')
    model = create_model(config_file['model_params'])
    model.setup()

    starting_epoch = config_file['model_params']['load_checkpoint'] + 1
    num_epochs = config_file['model_params']['epochs'] + 1

    for epoch in range(starting_epoch, num_epochs):
        epoch_start_time = time.time()
        train_dataset.dataset.pre_epoch_callback(epoch)
        model.pre_epoch_callback(epoch)

        model.train()
        for i, data in enumerate(train_dataset):
            model.set_input(data)
            model.forward()
            model.compute_loss()
            model.optimize_parameters()

        model.eval()
        for i, data in enumerate(val_dataset):
            model.set_input(data)
            model.test()

        model.post_epoch_callback(epoch)

        if epoch % config_file['save_freq'] == 0:
            logger.info(f'Saving model at the end of epoch {epoch}')
            model.save_networks(epoch)
            model.save_optimizers(epoch)

        model.update_learning_rate()

        logger.info(f'End of epoch {epoch} of {num_epochs - 1}')
        logger.info(f'Time taken: {time.time() - epoch_start_time:.3f} sec')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    with open('config_episodic_protonet.json') as json_file:
        configuration_file = json.load(json_file)

    # SageMaker env parameters
    if os.environ.get('SM_CHANNEL_TRAIN', None):
        configuration_file['model_params'][
            'checkpoint_path'] = os.environ['SM_OUTPUT_DATA_DIR']
        configuration_file[
            'train_dataset_params'
        ]['datapath'] = os.environ['SM_CHANNEL_TRAIN']
        configuration_file[
            'val_dataset_params'
        ]['datapath'] = os.environ['SM_CHANNEL_TEST']

    # Setup logger
    logger = duallog.setup(
        configuration_file['model_params']['checkpoint_path'],
        __name__,
        os.environ.get("LOGLEVEL", "INFO")
    )

    # TODO: load params via SageMaker if hyperparameter search

    train(configuration_file)
