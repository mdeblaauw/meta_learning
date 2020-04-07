import unittest
import os
import tempfile
import shutil
import sys

from utils.data_compression import tardir

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'few_shot'))

from few_shot.meta_learning.datasets.episodic_logo_dataset import \
    EpisodicLogoDataset


class test_data_compression(unittest.TestCase):
    """Test the compression and uncompress capability from,
    respectively, `utils/data_compression.py` and
    `few_shot/meta_learning/datasets/episodic_dataset.py`
    """
    def test_compression(self):
        temp_dir = tempfile.mkdtemp()

        #  Make dummy data structure
        file_path = os.path.join(temp_dir, 'images')
        # os.mkdir(file_path)
        # os.mkdir(os.path.join(file_path, 'train'))
        # os.mkdir(os.path.join(file_path, 'evaluate'))

        shutil.copytree('tests/dummy_datasets/images', file_path)

        # Create dummy tar file and test if it is there
        tardir(file_path, f'{file_path}.tar.gz', 'images')

        self.assertTrue(os.path.isfile(f'{file_path}.tar.gz'))

        # Test uncompress feature in `EpisodicDataset` class
        dummy_config = {
            'datapath': temp_dir,
            'subset': 'train',
            'resize': 30
        }

        dummy_dataset = EpisodicLogoDataset(dummy_config)

        # Check if `EpisodicDataset` has uncompressed corretly
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'train')))
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'evaluate')))

        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
