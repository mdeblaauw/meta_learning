import torch
from typing import List, Dict, Union
from torch.optim import lr_scheduler


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
