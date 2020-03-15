import torch
from typing import List, Union


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
