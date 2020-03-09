from typing import Dict, Tuple
from .episodic_dataset import EpisodicDataset
from PIL import Image
from torchvision import transforms


class EpisodicLogoDataset(EpisodicDataset):
    """This is a child class of the abstract base class for logo datasets.
    """
    def __init__(self, configuration: Dict,
                 transform: transforms.Compose = None):
        """Initialize the class, uses the configuration file to initialise the
        parent class and uses a transform pipeline if given.

        Arguments:
            configuration {Dict} -- Configuration file.
            transform {transforms.Compose} -- Image transformation pipeline.
        """
        super().__init__(configuration)
        self.transform = transforms.Compose(
            [transforms.Resize((configuration['resize'],
                                configuration['resize'])),
             transforms.ToTensor()]
        )
        if transform:
            self.transform = transform

    def read_image(self, image_path: str) -> Image:
        """Reads the image from disk.

        Arguments:
            image_path {str} -- Direct image path.

        Returns:
            PIL.Image -- A PIL Image between 2-4 dimensions
                (greyscal, rgb or rgba).
        """
        return Image.open(image_path).convert('RGB')

    def apply_transform(self, sample: Image) -> 'torch.FloatTensor':
        """Applies transformations. Standard, image is resized by given
        configuration parameter and the image is transformed to a FloatTensor
        in range [0.0, 1.0], see ToTensor @
        https://pytorch.org/docs/stable/torchvision/transforms.html

        Arguments:
            sample {PIL.Image} -- PIL Image.

        Returns:
            torch.FloatTensor -- A transformed sample
        """
        return self.transform(sample)

    def __getitem__(self, index: int) -> Tuple['torch.Tensor', int]:
        """Returns a sample, label data pair given a sample id.

        Arguments:
            index {int} -- Sample id.

        Returns:
            Tuple['torch.Tensor', int] -- Read and transformed image with
                label.
        """
        sample = self.read_image(self.datasetid_to_filepath[index])
        sample = self.apply_transform(sample)
        label = self.datasetid_to_class_id[index]

        return sample, label
