import os
import csv
import json
import io

from typing import Tuple
from collections import OrderedDict, Iterable
from .callback import Callback

import numpy as np
import torch


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