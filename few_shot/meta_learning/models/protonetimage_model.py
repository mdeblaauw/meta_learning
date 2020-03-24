from typing import Tuple, Dict
from .base_model import BaseModel
from ..utils import create_nshot_task, pairwise_distances, compute_class_mean
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
        nn.Conv2d(in_channels, out_channels, kernel=3, padding=1),
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
        # Length of encoding network.
        self.enc_size = [in_c, 64, 64, 64, 64]

        conv_blocks = [conv2_block(in_f, out_f) for
                       in_f, out_f in zip(self.enc_size, self.enc_size[:1])]

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


class ProtonetImage(BaseModel):
    def __init__(self, configuration: Dict):
        """Initialise the model.
        """
        super().__init__(configuration)

        self.loss_name = ['protonet']
        self.network_names['protonetstandard']

        self.protonet = ProtoNetStandard(3)
        self.protonet = self.protonet.to(self.device)

        if self.is_train:
            self.loss_function = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(
                self.protonet.parameters(), lr=configuration['lr']
            )
            self.optimizers = [self.optimizer]

        # Store predictions and labels.
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
        embeddings = self.protonet(self.input)

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

    def optimizer_parameters(self):
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

        self.val_predictions.append(self.input)
        self.val_labels.append(self.label)

    def post_epoch_callback(self, epoch: int):
        """Compute the batch average train accuracy and
        validation accuracy.

        Arguments:
            epoch {int} -- State of epoch.
        """
        self.train_predictions = torch.cat(self.train_predictions, dim=0)
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        self.train_labels = torch.cat(self.train_labels, dim=0)
        self.val_labels = torch.cat(self.val_labels, dim=0)

        train_accuracy = categorical_accuracy(
            self.train_predictions, self.train_labels
        )
        val_accuracy = categorical_accuracy(
            self.val_predictions, self.val_labels
        )

        print(f'Train accuracy: {train_accuracy:.3f}')
        print(f'Validation accuracy: {val_accuracy:.3f}')

        self.train_predictions = []
        self.val_predictions = []
        self.train_labels = []
        self.val_labels = []
