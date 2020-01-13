import os
import csv
import json
import io

from typing import Tuple
from collections import OrderedDict, Iterable
from .callback import Callback

import numpy as np
import torch


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