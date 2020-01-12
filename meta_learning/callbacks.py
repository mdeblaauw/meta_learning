import os
import csv
import json
import io
from typing import Tuple
from collections import OrderedDict, Iterable

import numpy as np
import torch

#from style_transfer.utils import save_image


class CallbackList(object):
    """
    Container abstracting a list of callbacks

    # Arguments
        callbacks: a list of Callback instances
    """
    def __init__(self, callbacks):
        self.callbacks = [c for c in callbacks]

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """
        Called at the end of training

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of an epoch

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """
        Called at the start of a batch

        # Arguments
            batch: integer, index of batch whithin current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch

        # Arguments
            batch: integer, index of batch whithin current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class Callback(object):
    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class MetricAggregator(Callback):
    """Callback that records metrics over epochs by taking the mean.

    """

    def on_epoch_begin(self, batch, logs=None):
        self.batches = 0
        self.totals = {}
        self.metrics = [self.params['loss']] + self.params['metrics']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        # Includes every key that contains the word loss from logs

        losses = [k for k in logs if 'loss' in k]
        self.metrics = self.metrics \
            + [loss for loss in losses if loss not in self.metrics]
        batch_size = logs.get('size', 1) or 1
        self.batches += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.metrics:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.batches


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
                print('Batch:', self.batch_iterations)

    def on_batch_end(self, batch, logs=None):
        if self.params['verbose'] == 2:
            if(self.batch_iterations % self.log_interval == 0):
                for k in self.metrics:
                    if logs[k]:
                        print(f'{k}:{logs[k]}')

    def on_epoch_end(self, epoch, logs=None):
        if self.params['verbose'] > 0:
            print('Epoch:', epoch)
        if self.params['verbose'] == 1:
            for k in self.metrics:
                if logs[k]:
                    print(f'{k}:{logs[k]}')

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


class SaveAsOnnx(Callback):
    """Callback that stores the model in ONNX format with dynamic height
        and width.

    # Arguments
        path {string} -- Path + filename to store the ONNX model.
                         Such as, `<path>/model.onnx`.
        dim_dummy {Tuple[int,int,int,int]} -- Dimension dummy
                                              with input (N,C,H,W).
    """

    def __init__(self, path: str, dim_dummy: Tuple[int, int, int, int]):
        self.path = path
        self.dummy_input = torch.randn(dim_dummy, requires_grad=True)
        if torch.cuda.is_available():
            self.dummy_input = self.dummy_input.to('cuda')
        super(SaveAsOnnx, self).__init__()

    def on_train_end(self, logs=None):
        self.model.eval()
        torch_out = self.model(self.dummy_input)
        torch.onnx.export(self.model, self.dummy_input, self.path,
                          verbose=False, input_names=['input'],
                          output_names=['output'], dynamic_axes={
                              'input': [2, 3], 'output': [2, 3]})


class SaveAsPytorch(Callback):
    """Callback that stores the model into a .pt extension with
        Pytorch's state_dict().

    Arguments:
        path {string} -- Path + filename to store the Pytorch model.
                         Such as, `<path>/model.pt`.
    """

    def __init__(self, path: str):
        self.path = path
        super(SaveAsPytorch, self).__init__()

    def on_train_end(self, logs=None):
        torch.save(self.model.state_dict(), self.path)


class SaveStyleTransferImage(Callback):
    """Callback that saves generated images

    Arguments:
        path {string} -- Path + filename to save image
        input_img {torch.Tensor} -- Image that should be saved
    """

    def __init__(self, path: str, input_img: torch.Tensor):
        self.path = path
        self.imageObject = input_img
        super(SaveStyleTransferImage, self).__init__()

    def on_train_end(self, logs=None):
        img = self.imageObject.clone()
        img = img.clamp(0, 1) * 255