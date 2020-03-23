from typing import Tuple
from .base_model import BaseModel
from ..utils import create_nshot_task, pairwise_distances, compute_class_mean
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
    def __init__(self, configuration):
        super().__init__(configuration)

        self.loss_name = ['protonet']
        self.network_names['protonetstandard']

        self.protonet = ProtoNetStandard(3)
        self.protonet = self.protonet.to(self.device)

        if self.is_train:
            self.loss_function = nn.NLLLoss()
            self.optimizer = torch.optim.Adam(
                self.protonet.parameters(), lr=configuration['lr']
            )
            self.optimizers = [self.optimizer]

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
        distances = pairwise_distances(queries, prototypes, self.distance)
        self.output = (-distances).softmax(dim=1)

    def compute_loss(self):

    def optimizer_parameters(self):

    def test(self):
    
