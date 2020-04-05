import logging.handlers
import os
import sys


def setup(log_directory: str, name: str, log_level: int = logging.WARNING):
    """This function setup dual logging, which means that logs
    are streamed to a log file and to the console.

    Arguments:
        log_directory {str} -- Directory in which log file(s) shoulde
            be stored.
        name {str} -- Logger name.

    Keyword Arguments:
        log_level {int} -- The level of logging (default: {logging.WARNING})
    """
    assert os.path.exists(log_directory), f'The {log_directory} does not exist'

    file_path = os.path.join(log_directory, 'output.log')

    logging.basicConfig(level=log_level,
                        handlers=[
                            logging.FileHandler(filename=file_path),
                            logging.StreamHandler(sys.stdout)
                        ])

    logger = logging.getLogger(name)

    return logger
