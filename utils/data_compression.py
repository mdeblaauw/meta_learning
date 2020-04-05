"""This script helps to compress data into a `.tar.gz` format. This is
especially handy if you are using SageMaker, which downloads data from S3.
The script will compress without including the parent folder. For example:

    you have the following data structure:
        `Logos/
            train/
                files
            evaluate/
                files`

    The script only includes the subchild folders of the Logos folder.
    In other words, the train and evaluate folders.
"""

import os
import tarfile
import argparse


def tardir(path: str, tar_name: str, folder: str):
    """Script that creates `tar.gz` file without including the parent folder.

    Arguments:
        path {str} -- Path to the parent folder.
        tar_name {str} -- Name of the compressed `tar.gz` file.
        folder {str} -- Folder name of the parent folder.
    """
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(
                    os.path.join(root, file),
                    arcname=root.split(f'/{folder}')[-1]
                )


if __name__ == '__main__':
    # Create parser for path input
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path-folder', type=str)
    args = parser.parse_args()

    # Execute compression script.
    # The compressed file is stored at the data path ...
    # with the folder as name.
    data_folder = os.path.basename(args.data_path_folder)
    tardir(
        args.data_path_folder,
        f'{args.data_path_folder}.tar.gz',
        data_folder
    )
