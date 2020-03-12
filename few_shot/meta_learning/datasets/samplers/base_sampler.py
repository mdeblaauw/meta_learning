from abc import ABC
from abc import abstractmethod
from typing import Dict
import torch.utils.data as data


class BaseSampler(data.Sampler, ABC):
    """This is an abstract base class for samplers.
    """
    def __init__(self, dataset: data.Dataset, configuration: Dict):
        """Initialize the class and save the configuration file in the class.

        Arguments:
            dataset {data.Dataset} -- Dataset from which to sampler from.
            configuration {Dict} -- Configuration file.
        """
        self.dataset = dataset
        self.configuration = configuration

    def __len__(self) -> int:
        """How many iterations per epoch the dataloader iterates.

        Returns:
            int -- Number of iterations per epoch.
        """
        return 0

    @abstractmethod
    def __iter__(self):
        """Logic in which samples per batch should be given.
        """
        pass
