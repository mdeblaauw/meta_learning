import torch
from typing import List, Dict, Union
from torch.optim import lr_scheduler

EPSILON = 1e-8


def transfer_to_device(x: Union[torch.Tensor, List, List[List]],
                       device: torch.device) -> Union[torch.Tensor,
                                                      List,
                                                      List[List]]:
    """Transfers torch tensors or lists to GPU. Or let them on CPU.
    This function is recursive. So, it can deal with lists of lists.

    Arguments:
        x {Union[torch.Tensor, List, List[List]]} -- Torch tensors that should
            be transferred to GPU.
        device {torch.device} -- Object that contains information on which
            device to store the tensor(s).

    Returns:
        Union[torch.Tensor, List, List[List]] -- Reference variables to
            transferred tensor(s).
    """
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = transfer_to_device(x[i], device)
    else:
        x = x.to(device)
    return x


def get_scheduler(optimizer: torch.optim,
                  configuration: Dict,
                  last_epoch=-1) -> torch.optim.lr_scheduler:
    """Return a learning rate scheduler.

    Arguments:
        optimizer {torch.optim} -- An optimizer object.
        configuration {Dict} -- The model parameters.

    Keyword Arguments:
        last_epoch {int} -- LR scheduler parameter that needs to set when
            resuming training from a checkpoint. (default: {-1})

    Returns:
        torch.optim.lr_scheduler -- A LR scheduler object.
    """
    if configuration['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=configuration['lr_decay_iters'],
            gamma=0.3, last_epoch=last_epoch
        )
    else:
        return NotImplementedError((f'Learning rate policy'
                                    f" {configuration['lr_policy']} "
                                    f'is not implemented'))
    return scheduler


def create_nshot_task(k: int, q: int) -> torch.Tensor:
    """Creates an n-shot task label based on episodic training.
    The label has the structure:
        [[0]*q, [1]*q, ..., [k-1]*q]

    Arguments:
        k {int} -- Number of classes in the n-shot classification task.
        q {int} -- Number of query samples for each class in the n-shot task.

    Returns:
        torch.Tensor -- Label tensor for n-shot task of shape [q * k].
    """
    return torch.arange(0, k, 1 / q).long()


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str,
                       S = None) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'gaussian':
        distances = ((
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2) * S.unsqueeze(0).expand(n_x, n_y,-1)).sum(dim=2)
    
        distances = distances.sqrt()
        return distances
    elif matching_fn == 'gaussian_v2':
        difference = x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1)
        difference_two = torch.matmul(S.unsqueeze(0).expand(n_x, n_y,-1,-1), difference.unsqueeze(3))
        distances = torch.matmul(difference.unsqueeze(2), difference_two).squeeze()
        distances = distances.sqrt()
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def compute_class_mean(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.
    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_means = support.reshape(k, n, -1).mean(dim=1)
    return class_means
