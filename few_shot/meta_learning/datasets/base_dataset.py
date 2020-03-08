from abc import ABC
from abc import abstractmethod
from typing import Dict
import torch.utils.data as data


class BaseDataset(data.Dataset, ABC):
    """This is an abstract base class for datasets.
    """
    def __init__(self, configuration: Dict):
        """Initialize the class and save the configuration file in the class.

        Arguments:
            configuration {Dict} -- Configuration file.
        """
        self.configuration = configuration

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int -- Total number of samples.
        """
        return 0

    @abstractmethod
    def __getitem__(self, index: int):
        """Return a sample. In a supervised setting, this would normally
        be a tuple that consist of a sample and label.

        Arguments:
            index {int} -- [description]
        """
        pass

    def pre_epoch_callback(self, epoch: int):
        """Callback to be called before an epoch.

        Arguments:
            epoch {int} -- The current epoch number.
        """
        pass

    def post_epoch_callback(self, epoch: int):
        """Callback to be called after an epoch.

        Arguments:
            epoch {int} -- The current epoch number
        """
        pass
