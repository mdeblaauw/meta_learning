import unittest
import torch
from few_shot.meta_learning.metrics import categorical_accuracy


class TestMetrics(unittest.TestCase):
    def test_categorical_accuracy(self):
        """Test the categorical_accuracy metric function.
        """
        dummy_input_pred = torch.Tensor([
            [0, 0, 0.5],
            [0.1, 1., 3.],
            [-1, -10, -20]
        ])

        dummy_input_y = torch.Tensor([2, 2, 0]).long()

        expected_output = 1.0
        output = categorical_accuracy(dummy_input_pred, dummy_input_y)

        self.assertEqual(expected_output, output)

    def test_categorical_accuracy_extended(self):
        """Test categorical_accuracy function with
        torch.cat, because it is used in protonet.
        """
        input_pred_1 = torch.Tensor([
            [0, 0.5],
            [0.1, 3.]
        ])
        label_1 = torch.Tensor([1, 1]).long()

        input_pred_2 = torch.Tensor([
            [0.1, 0.],
            [2., 10.]
        ])
        label_2 = torch.Tensor([0, 1]).long()

        input_preds = [input_pred_1, input_pred_2]
        labels = [label_1, label_2]

        cat_preds = torch.cat(input_preds, dim=0)
        cat_labels = torch.cat(labels, dim=0)

        expected_output = 1.0
        output = categorical_accuracy(cat_preds, cat_labels)

        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
