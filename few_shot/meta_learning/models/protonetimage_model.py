from typing import Tuple, Dict
from .base_model import BaseModel
from ..utils import create_nshot_task, pairwise_distances, \
    compute_class_mean, transfer_to_device
from ..metrics import categorical_accuracy
import torch
import torch.nn as nn


def conv2_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """A standard convolution block.

    Arguments:
        in_channels {int} -- Number of input channels.
        out_channels {int} -- Number of output channels.

    Returns:
        nn.Sequential -- A convolution block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to
    2-dimensional Tensor of shape [batch_size, d1*d2*...*dn].
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.size(0), -1)


class ProtoNetStandard(nn.Module):
    """Creates a standard embedding network introduced by Oriol Vinyals et al.
    See https://arxiv.org/pdf/1606.04080.pdf.
    """
    def __init__(self, in_c: int):
        """Initialisation of the model using input channel
        parameter.

        Arguments:
            in_c {int} -- Input channels. For RGB, three and
            greyscale, one.
        """
        super().__init__()
        # Length of encoding network.
        self.enc_size = [in_c, 64, 64, 64, 64]

        conv_blocks = [conv2_block(in_f, out_f) for
                       in_f, out_f in zip(self.enc_size, self.enc_size[1:])]

        self.encoding = nn.Sequential(*conv_blocks)
        self.representation = Flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of standard protonet.

        Arguments:
            input {torch.Tensor} -- Input data.

        Returns:
            torch.Tensor -- Output data.
        """
        x = self.encoding(input)
        return self.representation(x)


class ProtonetImageModel(BaseModel):
    def __init__(self, configuration: Dict):
        """Initialise the model.
        """
        super().__init__(configuration)

        self.k = configuration['k']
        self.n = configuration['n']
        self.q = configuration['q']
        self.distance = configuration['distance']

        self.loss_names = ['protonet']
        self.network_names = ['protonetstandard']

        self.netprotonetstandard = ProtoNetStandard(3)
        self.netprotonetstandard = self.netprotonetstandard.to(self.device)

        if self.is_train:
            self.loss_function = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(
                self.netprotonetstandard.parameters(), lr=configuration['lr']
            )
            self.optimizers = [self.optimizer]

        # Store predictions and labels.
        self.loss_value = []
        self.train_predictions = []
        self.val_predictions = []
        self.train_labels = []
        self.val_labels = []

    def set_input(self, input: Tuple[torch.Tensor, torch.Tensor]):
        """Unpack input data from DataLoader and perform pre-processing steps.

        Arguments:
            input {Tuple[torch.Tensor, torch.Tensor]} -- A tuple of input and
                label tensors.
        """
        self.input = transfer_to_device(input[0], self.device)
        self.label = transfer_to_device(
            create_nshot_task(self.k, self.q),
            self.device
        )

    def forward(self):
        """Run a forward pass for prototypical networks.
        """
        # Embed support and queries
        embeddings = self.netprotonetstandard(self.input)

        # Split support and queries.
        support = embeddings[:self.n*self.k]
        queries = embeddings[self.n*self.k:]

        # Compute prototypes for every support class
        prototypes = compute_class_mean(support, self.k, self.n)

        # Compute distances between queries and prototypes
        self.output = pairwise_distances(queries, prototypes, self.distance)

    def compute_loss(self):
        """Compute the loss.
        """
        self.loss_protonet = self.loss_function(-self.output, self.label)

        # temporary, until callbacklist is created.
        self.train_predictions.append((-self.output).softmax(dim=1))
        self.train_labels.append(self.label)
        loss = self.get_current_losses()
        self.loss_value.append(loss['protonet'])

    def optimize_parameters(self):
        """Perform a backward pass.
        """
        self.loss_protonet.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

    def test(self):
        """Perform testing.
        """
        # run a forward pass.
        super().test()

        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)

    def post_epoch_callback(self, epoch: int):
        """Compute the batch average train accuracy and
        validation accuracy.

        Arguments:
            epoch {int} -- State of epoch.
        """
        self.train_predictions = torch.cat(
            self.train_predictions, dim=0
        )
        self.val_predictions = (-1 * torch.cat(
            self.val_predictions, dim=0
        )).softmax(dim=1)
        self.train_labels = torch.cat(self.train_labels, dim=0)
        self.val_labels = torch.cat(self.val_labels, dim=0)

        train_accuracy = categorical_accuracy(
            self.train_predictions, self.train_labels
        )
        val_accuracy = categorical_accuracy(
            self.val_predictions, self.val_labels
        )

        print(f'Loss: {sum(self.loss_value)/len(self.loss_value)}')
        print(f'Train accuracy: {train_accuracy:.3f}')
        print(f'Validation accuracy: {val_accuracy:.3f}')

        self.loss_value = []
        self.train_predictions = []
        self.val_predictions = []
        self.train_labels = []
        self.val_labels = []
