import torch

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.

    This function calculates the precision of the output at the top k predictions for each class. 
    It returns a list of the precision values for each k in `topk`.

    Args:
        output (torch.Tensor): The output predictions from the model. Shape: (batch_size, num_classes).
        target (torch.Tensor): The true labels for the data. Shape: (batch_size,).
        topk (tuple of int): A tuple of integers specifying the values of k for which to compute the precision.

    Returns:
        list of float: A list of the precision values for each k in `topk`.
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    with torch.no_grad():
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]