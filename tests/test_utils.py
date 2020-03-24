import unittest
import mock
import math
import torch
from few_shot.meta_learning.utils import \
    transfer_to_device, get_scheduler, create_nshot_task, compute_class_mean, \
    pairwise_distances


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

    @mock.patch('few_shot.meta_learning.utils.lr_scheduler')
    def test_get_scheduler(self, mock_lr_scheduler):
        """Test if different conditions are called in get_scheduler.

        Arguments:
            mock_lr_scheduler -- Mock torch.optim.lr_scheduler.
        """
        output = get_scheduler('a', {'lr_policy': 'a'})
        self.assertIsInstance(output, NotImplementedError)

        # test if step calls StepLR
        mock_lr_scheduler.StepLR.return_value = 'called'
        output = get_scheduler(
            'a', {'lr_policy': 'step', 'lr_decay_iters': 'b'}
        )
        self.assertEqual(output, 'called')

    def test_create_n_shot_task(self):
        """Test label creation for episodic training. For this test,
        we mock a k=3 and q=2 case.
        """
        expected_output = torch.Tensor([0, 0, 1, 1, 2, 2]).long()

        output = create_nshot_task(k=3, q=2)

        self.assertTrue(torch.all(output.eq(expected_output)))

    def test_compute_class_mean(self):
        """Test compute class mean function.
        """
        dummy_input = torch.Tensor([
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [3, 3, 3],
            [3, 3, 3]
        ])

        expected_output = torch.Tensor([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])

        output = compute_class_mean(dummy_input, k=3, n=2)

        self.assertTrue(torch.all(output.eq(expected_output)))

    def test_pairwise_distances(self):
        """Test the pairwise_distances function on different similarity
        functions and raises.
        """
        # Test l2 (euclidian distance)
        l2_x_input = torch.Tensor([
            [1, 1],
            [3, 3]
        ])

        l2_y_input = torch.Tensor([
            [3, 3],
            [1, 1]
        ])

        l2_expected_output = torch.Tensor([
            [2 * math.sqrt(2), 0],
            [0, 2 * math.sqrt(2)]
        ])

        output_l2 = pairwise_distances(l2_x_input, l2_y_input, 'l2')
        self.assertTrue(torch.all(output_l2.eq(l2_expected_output)))

        # Test gaussian distance
        gaussian_S_input = torch.Tensor([
            [1, 1],
            [2, 2]
        ])

        gaussian_expected_output = torch.Tensor([
            [2 * math.sqrt(2), 0],
            [0, 4]
        ])

        output_gaussian = pairwise_distances(
            l2_x_input, l2_y_input, 'gaussian', gaussian_S_input
        )
        self.assertTrue(
            torch.all(output_gaussian.eq(gaussian_expected_output))
        )

        # Test cosine distance
        cosine_expected_output = torch.Tensor([
            [0., 0.],
            [0., 0.]
        ])

        output_cosine = pairwise_distances(
            l2_x_input, l2_y_input, 'cosine'
        )
        self.assertTrue(
            torch.allclose(output_cosine, cosine_expected_output, atol=1e-5)
        )

        # Test dot product distance
        dot_expected_output = torch.Tensor([
            [-6, -2],
            [-18, -6]
        ])

        output_dot = pairwise_distances(
            l2_x_input, l2_y_input, 'dot'
        )
        self.assertTrue(
            torch.all(output_dot.eq(dot_expected_output))
        )

        self.assertRaises(
            ValueError, pairwise_distances, l2_x_input, l2_y_input, 'gaussian'
        )
        self.assertRaises(
            ValueError, pairwise_distances, l2_x_input, l2_y_input, 'dummy'
        )


if __name__ == '__main__':
    unittest.main()
