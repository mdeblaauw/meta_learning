import unittest

from few_shot.meta_learning.datasets.episodic_dataset import \
    EpisodicDataset
from few_shot.meta_learning.datasets.episodic_image_dataset import \
    EpisodicImageDataset


class test_datasets(unittest.TestCase):
    def test_episodic_dataset(self):
        image_path = 'tests/dummy_datasets/images'
        configuration = {}
        subset = 'train'

        image_ep_data = EpisodicImageDataset(configuration, subset, image_path)

        numb_samples = 10
        numb_classes = 3

        self.assertEqual(len(image_ep_data), numb_samples)
        self.assertEqual(image_ep_data.num_classes(), numb_classes)

    def test_episodic_image_dataset(self):
        image_path = 'tests/dummy_datasets/images'
        configuration = {}
        subset = 'train'

        image_ep_data = EpisodicImageDataset(configuration, subset, image_path)


if __name__ == '__main__':
    unittest.main()
