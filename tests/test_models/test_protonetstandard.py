import unittest
import torch
from few_shot.meta_learning.models.protonetimage_model import \
    ProtoNetStandard


class TestProtoNetStandardModel(unittest.TestCase):
    def test_protnetstandard_architecture(self):
        """Test the number of layers.
        """
        test_model = ProtoNetStandard(3)

        self.assertEqual(len(test_model.encoding), 4)


if __name__ == '__main__':
    unittest.main()
