import os
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Dict
from collections import OrderedDict
from ..utils import transfer_to_device, get_scheduler


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    """
    def __init__(self, configuration: Dict):
        """Initialise the BaseModel class.

        Arguments:
            configuration {Dict} -- The configuration file.

        When creating your own custom class, you need to implement a custom
        initialisation. In this function, you should first call
        `super().__init__(self, configuration)`. Then, you need to define
        these lists:
            -- self.network_names {List[str]}. Should be `net[name]`.
            -- self.loss_names {List[str]}. Should be `loss_[name]`
            -- self.optimizers {List[torch.optim]}
        """
        self.configuration = configuration
        self.is_train = configuration['is_train']
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') \
            if self.use_cuda else torch.device('cpu')
        # only true when same input size to model for every batch.
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
        self.input = transfer_to_device(input[0], self.device)
        self.label = transfer_to_device(input[1], self.device)

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
    def optimize_parameters(self):
        """Calculate gradients and update network weights;
        called in every training iteration.
        """
        pass

    def setup(self):
        """Load and print networks. Also, create schedulers.
        Load_checkpoint higher than 0 means that it tries to load
        a previous checkpoint at epoch equal load_checkpoint. A load_checkpoint
        of zero means a fresh training model without loading.
        """
        if self.configuration['load_checkpoint'] > 0:
            last_checkpoint = self.configuration['load_checkpoint']
        else:
            last_checkpoint = 0

        if last_checkpoint > 0:
            # enable restarting training
            self.load_networks(last_checkpoint)
            if self.is_train:
                self.load_optimizers(last_checkpoint)

        self.schedulers = [get_scheduler(optimizer, self.configuration) for
                           optimizer in self.optimizers]

        if last_checkpoint > 0:
            for s in self.schedulers:
                for i in range(last_checkpoint):
                    s.step()

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
        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'Learning rate = {lr}')

    def save_networks(self, epoch: int):
        """Save all the networks to disk.

        Arguments:
            epoch {int} -- State of epoch.
        """
        for name in self.network_names:
            if isinstance(name, str):
                save_filename = f'{epoch}_net_{name}.pth'
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.use_cuda:
                    torch.save(net.cpu(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch: int):
        """Load all the networks from disk.

        Arguments:
            epoch {int} -- State of epoch.
        """
        for name in self.network_names:
            if isinstance(name, str):
                load_filename = f'{epoch}_net_{name}.pth'
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net.module
                print(f'Loading the model from {load_path}')
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def save_optimizers(self, epoch: int):
        """Save all the optimizers to disk. Such that
        you can restart training.

        Arguments:
            epoch {int} -- State of epoch.
        """
        for i, optimizer in enumerate(self.optimizers):
            save_filename = f'{epoch}_optimizer_{i}.pth'
            save_path = os.path.join(self.save_dir, save_filename)

            torch.save(optimizer.state_dict(), save_path)

    def load_optimizers(self, epoch: int):
        """Load all the optimizers from disk.

        Arguments:
            epoch {int} -- State of epoch.
        """
        for i, optimizer in enumerate(self.optimizers):
            load_filename = f'{epoch}_optimizer_{i}.pth'
            load_path = os.path.join(self.save_dir, load_filename)
            print(f'Loading the optimizer from {load_path}')
            state_dict = torch.load(load_path)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            optimizer.load_state_dict(state_dict)

    def print_networks(self):
        """Print the total number of parameters in the network and network
        architecture.
        """
        print('Network initialized')
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print((f'Network {name}, has a total number of '
                       f'parameters: {num_params}'))

    def get_current_losses(self) -> float:
        """Return training losses. train.py prints these out.

        Returns:
            float -- Computed loss.
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
            return errors_ret

    def pre_epoch_callback(self, epoch: int):
        pass

    def post_epoch_callback(self, epoch: int):
        pass
