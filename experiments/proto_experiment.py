import argparse

import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from meta_learning.models.models import get_few_shot_encoder
from meta_learning.train_iterators.protonet import *
from meta_learning.core import *
from meta_learning.datasets.datasets import OmniglotDataset
from meta_learning.train import fit
from meta_learning.callbacks import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Terminal input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--distance", type=str,
                    default="l2",
                    help="")
parser.add_argument("--n-train", type=int, default=5,
                    help="")
parser.add_argument("--k-train", type=int, default=5,
                    help="")
parser.add_argument("--q-train", type=int, default=5,
                    help="")
parser.add_argument("--n-test", type=int, default=5,
                    help="")
parser.add_argument("--k-test", type=int, default=5,
                    help="")
parser.add_argument("--q-test", type=int, default=5,
                    help="")
parser.add_argument("--evaluation-episodes", type=int, default=20,
                    help="")
parser.add_argument("--episodes-per-epoch", type=int, default=5,
                    help="")
parser.add_argument("--final-evaluation-episodes", type=int, default=5,
                    help="")
parser.add_argument("--epochs", type=int, default=5,
                    help="")
parser.add_argument("--dataset-name", type=str, default='Omniglot',
                    help="")
parser.add_argument("--num-input-channels", type=int, default=1,
                    help="")
parser.add_argument("--learning-rate", type=float, default=0.001,
                    help="")

args = parser.parse_args()

dataset_class = OmniglotDataset
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(
        background,
        args.episodes_per_epoch,
        args.n_train,
        args.k_train,
        args.q_train
    ),
    num_workers=4
)

evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(
        evaluation,
        args.evaluation_episodes,
        args.n_test,
        args.k_test,
        args.q_test
    ),
    num_workers=4
)

model = get_few_shot_encoder(
    args.num_input_channels
).to(device, dtype=torch.double)

optimiser = Adam(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.NLLLoss()

callbacks = [VerboseLogger()]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=args.epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    metrics=['categorical_accuracy'],
    callbacks=callbacks,
    verbose=1,
    fit_function=proto_net_episode,
    fit_function_kwargs={'n_shot':args.n_train,
                         'k_way':args.k_train,
                         'q_queries':args.q_train,
                         'distance':args.distance,
                         'train':True}
)