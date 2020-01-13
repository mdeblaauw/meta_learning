import os
import csv
import json
import io
import logging
import sys

from typing import Tuple
from collections import OrderedDict, Iterable
from .callback import Callback

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class VerboseLogger(Callback):
    """Callback that prints training information to the terminal.
       For example, epoch number and corresponding average loss.

    # Arguments
        log_interval {int} -- If verbose is 2, then this sets the number of
                              images after which the training loss is logged
                              (default {1})
    """

    def __init__(self, log_interval: int = 1):
        self.batch_iterations = 0
        self.log_interval = log_interval

    def on_batch_begin(self, batch, logs=None):
        if self.params['verbose'] == 2:
            self.batch_iterations += 1
            if(self.batch_iterations % self.log_interval == 0):
                #print('Batch:', self.batch_iterations)
                logger.info(f'Batch: {self.batch_iterations}')

    def on_batch_end(self, batch, logs=None):
        if self.params['verbose'] == 2:
            if(self.batch_iterations % self.log_interval == 0):
                for k in self.metrics:
                    if logs[k]:
                        #print(f'{k}:{logs[k]}')
                        logger.info(f'{k}:{logs[k]}')

    def on_epoch_end(self, epoch, logs=None):
        if self.params['verbose'] > 0:
            #print('Epoch:', epoch)
            logger.info(f'Epoch: {epoch}')
        if self.params['verbose'] == 1:
            for k in self.metrics:
                if logs[k]:
                    #print(f'{k}:{logs[k]}')
                    logger.info(f'{k}:{logs[k]}')

    def on_train_begin(self, logs=None):
        self.metrics = [self.params['loss']] + self.params['metrics']
        if self.params['verbose'] > 0:
            print('Begin training...')

    def on_train_end(self, logs=None):
        if self.params['verbose'] > 0:
            print('Finished.')


class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        self.iterations = 0
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'

        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_batch_end(self, batch, logs=None):
        if self.params['verbose'] == 2:
            logs = logs or {}
            self.iterations += 1

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, str):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['batch_iteration'] + self.keys
                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = OrderedDict({'batch_iteration': self.iterations})
            row_dict.update((key, handle_value(logs[key])) for key in
                            self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

    def on_epoch_end(self, epoch, logs=None):
        if self.params['verbose'] != 2:
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, str):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['epoch'] + self.keys
                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = OrderedDict({'epoch': epoch})
            row_dict.update((key, handle_value(logs[key])) for key in
                            self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class HypoLogger(Callback):
    """Callback that stores the hyperparameters in a json.

    # Arguments
        path: path with filename where hyperparameters are written to
        arg_hypos: dictionary with argparse variables
        **kwargs: any other hyperparameters given to the callback.
        Can be strings or classes.
    """

    def __init__(self, path, arg_hypos, **kwargs):
        self.path = path
        self.arg_hypos = arg_hypos
        self.kwargs = kwargs
        super(HypoLogger, self).__init__()

    def on_train_begin(self, logs=None):
        self.arg_hypos['model'] = self.model.__class__.__name__
        self.arg_hypos['optimiser'] = \
            self.params['optimiser'].__class__.__name__
        self.arg_hypos['loss_fn'] = self.params["loss_fn"].__class__.__name__
        if self.kwargs:
            for key, value in self.kwargs.items():
                if isinstance(value, object):
                    self.arg_hypos[key] = value.__class__.__name__
                else:
                    self.arg_hypos[key] = value

        with open(self.path, 'w') as fp:
            json.dump(self.arg_hypos, fp)