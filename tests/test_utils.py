import unittest
import torch
from few_shot.meta_learning.utils import transfer_to_device


class TestUtils(unittest.TestCase):

    def test_transfer_to_device(self):
        """Test if function runs with different input and if it
        gives correct output.
        """
        dummy_torch = [torch.Tensor([1, 2]), torch.Tensor([3, 4])]
        dummy_torch_v2 = [[torch.Tensor([1, 2])], [torch.Tensor([3, 4])]]

        out = transfer_to_device(dummy_torch, torch.device("cpu"))
        for i, j in zip(dummy_torch, out):
            self.assertTrue(torch.all(i.eq(j)))

        out = transfer_to_device(dummy_torch_v2, torch.device("cpu"))
        for i, j in zip(dummy_torch_v2, out):
            self.assertTrue(torch.all(i[0].eq(j[0])))


if __name__ == '__main__':
    unittest.main()
