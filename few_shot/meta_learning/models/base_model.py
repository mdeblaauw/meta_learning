import os
import torch
from abc import ABC, abstractmethod
from typing import Tuple


class BaseModel(ABC):
    def __init__(self, configuration):
        self.configuration = configuration
        self.is_train = configuration['is_train']
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') \
            if self.use_cuda else torch.device('cpu')
        # only true when same input size to model.
        torch.backends.cudnn.benchmark = True
        self.save_dir = configuration['checkpoint_path']
        self.network_names = []
        self.loss_names = []
        self.optimizers = []

    def set_input(self, input: Tuple[torch.Tensor, torch.Tensor]):
        self.input = transfer_to_device(self.input[0], self.device)
        self.label = transfer_to_device(self.input[1], self.device)

    @abstractmethod
    def forward(self):

    @abstractmethod
    def compute_loss(self):

    @abstractmethod
    def optimizer_parameters(self):

    def train(self):

    def eval(self):

    def test(self):

    def update_learning_rate(self):

    def save_networks(self, epoch):

    def load_networks(self, epoch):

    def save_optimizers(self, epoch):

    def load_optimizers(self, epoch):

    def print_networks(self):

    def set_requires_grad(self):

    def get_current_losses(self):

    def pre_epoch_callback(self, epoch):

    def post_epoch_callback(self, epoch):
