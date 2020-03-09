import importlib
from meta_learning.datasets.base_dataset import BaseDataset
from meta_learning.datasets.samplers import check_for_sampler, create_sampler
from typing import Dict
from torch.utils import data


def find_dataset_using_name(dataset_name: str):
    """Import the module meta_learning/datasets/[dataset_name]_dataset.py.

    Arguments:
        dataset_name {str} -- Prefix of module name and name for class
            instantiation, which is DatasetNameDataset().

    Raises:
        NotImplementedError: whether DatasetNameDataset() is implemented in
            [dataset_name]_dataset.py

    Returns:
        cls -- Class DatasetNameDataset() from
            [dataset_name]_dataset.py.
    """
    dataset_filename = f'meta_learning.datasets.{dataset_name}_dataset'
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            f'''In {dataset_filename}, there should be a subclass of
            BaseDataset with class that matches {target_dataset_name} in
            lowercase'''
        )

    return dataset


def create_dataset(configuration: Dict) -> 'CustomDatasetDataLoader':
    """Create a datasetloader given the configuration. This function
    wraps the CustomDatasetLoader, which is the main interface between
    train.py and validate.py.

    Returns:
        CustomDatasetDataLoader -- An instantiated custom dataloader.
    """
    data_loader = CustomDatasetDataLoader(configuration)
    return data_loader


class CustomDatasetDataLoader():
    """Wrapper class of DataLoader.
    """
    def __init__(self, configuration: Dict):
        """Initialise the parameters with configuration and create
        a torch DataLoader.

        Arguments:
            configuration {Dict} -- Configuration file.
        """
        self.configuration = configuration
        dataset_class = find_dataset_using_name(configuration['dataset_name'])
        self.dataset = dataset_class(configuration)
        print(f'Dataset {type(self.dataset).__name__} is created')

        # A custom sampler is used if there exist one for the dataset
        if check_for_sampler(configuration['dataset_name']):
            self.sampler = create_sampler(configuration)
            print(f'Sampler {self.sampler.__name__} is created')

            self.dataloader = data.DataLoader(
                self.dataset, **configuration['loader_params'],
                sampler=self.sampler
            )
        else:
            self.dataloader = data.DataLoader(
                self.dataset, **configuration['loader_params']
            )

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int -- Total number of samples.
        """
        return len(self.dataset)

    def __iter__(self) -> 'Generator':
        """Returns a batch of data.
        """
        for data in self.dataloader:
            yield data
