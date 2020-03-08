from typing import Dict
from .episodic_dataset import EpisodicDataset
import numpy as np
from skimage import io
from torchvision import transforms


class EpisodicImageDataset(EpisodicDataset):
    """This is a child class of the abstract base class for image datasets.
    """
    def __init__(self, configuration: Dict,
                 subset: str, datapath: str,
                 transform: transforms.Compose = None):
        """Initialize the class, uses the configuration file to initialise the
        parent class and uses a transform pipeline if given.

        Arguments:
            configuration {Dict} -- Configuration file.
            transform {transforms.Compose} -- Image transformation pipeline.
        """
        super().__init__(configuration, datapath, subset)
        # TODO resize transform
        self.transform = transforms.Compose([transforms.ToTensor()])
        if transform:
            self.transform = transform

    def read_image(self, image_path: str) -> np.ndarray:
        """Reads the image from disk.

        Arguments:
            image_path {str} -- Direct image path.

        Returns:
            np.ndarray -- A numpy matrix between 2-4 dimensions
                (greyscal, rgb or rgba).
        """
        return io.imread(image_path)

    def apply_transform(self, sample: np.ndarray) -> 'torch.FloatTensor':
        """Applies transformations. Standard, image is resized by given
        configuration parameter and the image is transformed to a FloatTensor
        in range [0.0, 1.0], see ToTensor @
        https://pytorch.org/docs/stable/torchvision/transforms.html

        Returns:
            torch.FloatTensor -- A transformed sample
        """
        return self.transform(sample)

    def __getitem__(self, index: int):
        return 0
