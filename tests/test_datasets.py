import unittest
import torch

from few_shot.meta_learning.datasets.episodic_dataset import \
    EpisodicDataset
from few_shot.meta_learning.datasets.episodic_image_dataset import \
    EpisodicImageDataset


class test_datasets(unittest.TestCase):
    def test_episodic_dataset(self):
        image_path = 'tests/dummy_datasets/images'
        configuration = {'resize': 50}
        subset = 'train'

        image_ep_data = EpisodicImageDataset(configuration, subset, image_path)

        numb_samples = 10
        numb_classes = 3

        self.assertEqual(len(image_ep_data), numb_samples)
        self.assertEqual(image_ep_data.num_classes(), numb_classes)

    def test_episodic_image_dataset(self):
        image_path = 'tests/dummy_datasets/images'
        configuration = {'resize': 50}
        subset = 'train'

        image_ep_data = EpisodicImageDataset(configuration, subset, image_path)
        sample, label = image_ep_data[0]

        self.assertEqual(label, 2)
        self.assertEqual(type(label), int)
        self.assertEqual(type(sample), torch.Tensor)
        self.assertEqual(sample.type(), 'torch.FloatTensor')
        # Check resize transform to 50
        self.assertEqual(sample.shape[1], 50)


if __name__ == '__main__':
    unittest.main()
