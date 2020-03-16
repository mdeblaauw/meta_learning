import os
import torch
from abc import ABC, abstractmethod
from typing import Tuple
from ..utils import transfer_to_device, get_scheduler


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
        """Unpack input data from DataLoader and perform pre-processing steps.
        This is a basic implementation. You can implement other functionalities
        for different models.

        Arguments:
            input {Tuple[torch.Tensor, torch.Tensor]} -- A tuple of input and
                label tensors.
        """
        self.input = transfer_to_device(self.input[0], self.device)
        self.label = transfer_to_device(self.input[1], self.device)

    @abstractmethod
    def forward(self):
        """Run a forward pass through the models;
        called in every training iteration.
        """
        pass

    @abstractmethod
    def compute_loss(self):
        """Compute the losses from the output of the forward pass;
        called in every training iteration.
        """
        pass

    @abstractmethod
    def optimizer_parameters(self):
        """Calculate gradients and update network weights;
        called in every training iteration.
        """
        pass

    def setup(self):
        """Load and print networks. Also, create schedulers. TODO
        """
        if self.configuration['load_checkpoint'] >= 0:
            print('Not yet implemented!')
        else:
            last_checkpoint = -1

        self.schedulers = [get_scheduler(optimizer, self.configuration) for
                           optimizer in self.optimizers]

        # self.print_networks()

    def train(self):
        """Set models in train mode.
        """
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def eval(self):
        """Set models in eval mode.
        """
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function to use at test time.
        This function wraps `self.forward()` function in torch.no_grad().
        So that intermediate steps are not saved for backprop.
        """
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self):
        """Update learning rates; called at the end of every epoch.
        """
        for scheduler in self.schedulers:
            scheduler.step()

        # Print new learning rate.
        lr = self.optimizer[0].param_groups[0]['lr']
        print(f'Learning rate = {lr}')

    def save_networks(self, epoch):

    def load_networks(self, epoch):

    def save_optimizers(self, epoch):

    def load_optimizers(self, epoch):

    def print_networks(self):

    def set_requires_grad(self):

    def get_current_losses(self):

    def pre_epoch_callback(self, epoch):

    def post_epoch_callback(self, epoch):
