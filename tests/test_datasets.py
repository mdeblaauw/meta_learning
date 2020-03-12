import unittest
import torch
import sys
import os
import mock
from unittest.mock import MagicMock, PropertyMock
from torch.utils import data

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'few_shot'))

from few_shot.meta_learning.datasets.episodic_dataset import \
    EpisodicDataset
from few_shot.meta_learning.datasets.episodic_logo_dataset \
    import EpisodicLogoDataset
from few_shot.meta_learning.datasets import create_dataset
from few_shot.meta_learning.datasets import find_dataset_using_name
from few_shot.meta_learning.datasets.samplers import \
    check_for_sampler, create_sampler
from few_shot.meta_learning.datasets.samplers.episodic_sampler import \
    EpisodicSampler


class test_datasets(unittest.TestCase):
    def test_episodic_dataset(self):
        configuration = {'datapath': 'tests/dummy_datasets/images',
                         'resize': 50, 'subset': 'train'}

        logo_ep_data = EpisodicLogoDataset(configuration)

        numb_samples = 10
        numb_classes = 3

        self.assertEqual(len(logo_ep_data), numb_samples)
        self.assertEqual(logo_ep_data.num_classes(), numb_classes)

    def test_episodic_logo_dataset(self):
        configuration = {'datapath': 'tests/dummy_datasets/images',
                         'resize': 50, 'subset': 'train'}

        logo_ep_data = EpisodicLogoDataset(configuration)
        sample, label = logo_ep_data[0]

        self.assertEqual(label, 2)
        self.assertEqual(type(label), int)
        self.assertEqual(type(sample), torch.Tensor)
        self.assertEqual(sample.type(), 'torch.FloatTensor')
        # Check resize transform to 50
        self.assertEqual(sample.shape[1], 50)
        self.assertEqual(sample.shape[2], 50)
        self.assertEqual(sample.shape[0], 3)

    def test_find_dataset_using_name(self):
        """Unittest find_dataset_using_name method.
        """
        dataset = find_dataset_using_name('episodic_logo')

        self.assertEqual(dataset.__name__, 'EpisodicLogoDataset')

    @mock.patch("few_shot.meta_learning.datasets.importlib")
    def test_find_dataset_using_name_raises(self, mock_importlib):
        """Test NotImplementedError raise by mocking importlib.
        """
        mock_importlib.import_module

        self.assertRaises(
            NotImplementedError, find_dataset_using_name, 'episodic_dummy'
        )

    def test_create_dataset(self):
        """Test if dataloader is returned with sampler and is able to iterate
        one epoch through samples.
        """
        configuration = {'datapath': 'tests/dummy_datasets/images',
                         'resize': 50, 'subset': 'train',
                         'dataset_name': 'episodic_logo',
                         'loader_params': {
                                           "num_workers": 0,
                                           "pin_memory": False
                                },
                         'sampler_params': {
                                            "episodes_per_epoch": 2,
                                            "k": 1,
                                            "n": 1,
                                            "q": 1
                                }
                         }

        data_loader = create_dataset(configuration)
        self.assertEqual(len(data_loader), 10)

        batch = data_loader

        for i, data in enumerate(data_loader):
            last = i
            x, y = data
        self.assertEqual(last, 1)
        self.assertEqual(x.size(), torch.Size([2, 3, 50, 50]))
        self.assertEqual(y.size(), torch.Size([2]))

    def test_check_for_sampler(self):
        input_string = 'episodic_logo'
        output = check_for_sampler(input_string)
        self.assertTrue(output)

        output = check_for_sampler('false')
        self.assertFalse(output)

    # def test_episodic_sampler(self):
    #     configuration = {'datapath': 'tests/dummy_datasets/images',
    #                      'resize': 50, 'subset': 'train',
    #                      'dataset_name': 'episodic_logo',
    #                      'loader_params': {
    #                                        "num_workers": 0,
    #                                        "pin_memory": False
    #                             },
    #                      'sampler_params': {
    #                                         "episodes_per_epoch": 10,
    #                                         "k": 1,
    #                                         "n": 1,
    #                                         "q": 1
    #                             }
    #                      }

    #     logo_ep_data = EpisodicLogoDataset(configuration)

    #     sampler = EpisodicSampler(logo_ep_data, configuration['sampler_params'])
    #     # out_iter = iter(sampler)
    #     # print(next(out_iter))
    #     loder = data.DataLoader(dataset=logo_ep_data, batch_sampler=sampler)
    #     for i in enumerate(loder):
    #         print(i)


if __name__ == '__main__':
    unittest.main()
