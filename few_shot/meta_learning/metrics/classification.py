import torch


def categorical_accuracy(y_pred: torch.Tensor,
                         y: torch.Tensor) -> float:
    """Calculates categorical accuracy.

    Arguments:
        y_pred {torch.Tensor} -- Prediction probabilities or logits of shape
            [batch_size, num_categories].
        y {torch.Tensor} -- Ground truth categories. Must have shape
            [batch_size].

    Returns:
        float -- Accuracy score.
    """
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]
