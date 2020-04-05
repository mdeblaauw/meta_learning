import os
import logging

from typing import List, Dict
from .base_dataset import BaseDataset

from tqdm import tqdm
import pandas as pd


class EpisodicDataset(BaseDataset):
    """This is a child class of the base dataset class. For which
    this class implements the structure for episodic training.
    See the static index subset method to see how the data should be
    structured.
    """
    def __init__(self, configuration: Dict, preprocess_fn: str = None):
        """Initialise the parameters and the BaseDataset parent class.

        Arguments:
            configuration {Dict} -- Configuration file.
            datapath {str} -- Path to train and evaluation folders.
            subset {str} -- Whether data should be created from train
                or evaluation data.

        Keyword Arguments:
            preprocess_fn {str} -- TODO: Which function to preprocess
                the data, such as unzipping. (default: {None})
        """
        assert configuration['subset'] in ['train', 'evaluate'], '''Subset must
                                            be one of train" or "evaluate'''

        super().__init__(configuration)
        self.datapath = configuration['datapath']
        self.subset = configuration['subset']
        self.preprocess_fn = preprocess_fn

        if preprocess_fn:
            # do some preprocessing
            dummy

        self.df = pd.DataFrame(self.index_subset(self.subset, self.datapath))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {
            self.unique_characters[i]: i for i in range(self.num_classes())
        }
        self.df = self.df.assign(
            class_id=self.df['class_name'].apply(
                lambda c: self.class_name_to_id[c]
            )
        )

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int -- Total number of samples.
        """
        return len(self.df)

    def num_classes(self) -> int:
        """Return the total number of classes.

        Returns:
            int -- Total number of classes.
        """
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset: str, datapath: str) -> List[Dict]:
        """Creates a list with information of every sample by looping
        through every file given by the datapath and subset.
        It is expected to have the following structure:

        datapath/
            subset/
                class_a/
                    file1
                    file2
                    .
                    .
                    fileN
                .
                .
                class_M/

        # Arguments
            subset: Name of the subset.

        # Returns
            A list of dicts containing information about every file.
                Which is subset, class_name and filepath.
        """
        samples = []
        logging.info(f'Indexing {subset}...')
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(
                os.path.join(datapath, subset)):
            subset_len += len(
                [f for f in files]
            )

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(
                os.path.join(datapath, subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                samples.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return samples
