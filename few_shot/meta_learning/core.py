from typing import List, Iterable, Callable, Tuple
import numpy as np
import torch


def create_nshot_task_label(k, q):
    """Creates an n-shot task label.
    Label has the structure:
        [0]*q + [1]*q + ... + [k-1]*q
    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task
    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(0, k, 1 / q).long()
    return y

def prepare_nshot_task(n: int, k: int, q: int, cuda: bool=False) -> Callable:
    """Typical n-shot task preprocessing.
        Note: this is a nested function utilising closure in Python.
    # Arguments
        n: Number of samples for each class in the n-shot classification task
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task
        cuda: Decision to use cuda
    # Returns
        prepare_nshot_task_: A Callable that processes a few shot tasks with specified n, k and q
    """
    def prepare_nshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create 0-k label and move to GPU.
        TODO: Move to arbitrary device
        """
        x, y = batch
        if cuda:
            x = x.double().cuda()
            # Create dummy 0-(num_classes - 1) label
            y = create_nshot_task_label(k, q).cuda()
        else:
            x = x
            y = create_nshot_task_label(k, q)
        return x, y

    return prepare_nshot_task_
