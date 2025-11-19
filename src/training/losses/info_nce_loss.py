import torch
from torch.nn import functional as F
from torch import nn
# from ot.lp import wasserstein_1d, wasserstein_circle
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, Function
from torchvision import datasets, transforms


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='paired', sim_func='cosine'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.sim_func = sim_func
        print("Sim function is " + sim_func)
        print("Negative mode is " + negative_mode)

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys, temperature=self.temperature, reduction=self.reduction, negative_mode=self.negative_mode, sim_func=self.sim_func)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='paired', sim_func='cosine'):
    """
    Calculates the InfoNCELoss loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor. M is the number of negative keys per sample.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the info_nce loss function.

    This function was adapted from https://github.com/RElbers/info-nce-pytorch/blob/main/InfoNCELoss/__init__.py
    """
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        if negative_mode == 'unpaired' and sim_func == 'cosine':
            # Cosine between positive pairs
            positive_logit = cos_similarity(query, positive_key, mode='paired')
            # Cosine between all query-negative combinations
            negative_logits = cos_similarity(query, negative_keys, mode='unpaired')

        elif negative_mode == 'paired' and sim_func == 'cosine':
            # Cosine between positive pairs
            positive_logit = cos_similarity(query, positive_key, mode='paired')
            # Cosine between all query-negative combinations
            negative_logits = cos_similarity(query, negative_keys, mode='paired')

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def cos_similarity(query, keys, mode='paired'):
    if len(keys.shape) == 2 and mode == 'paired':
        logits = torch.sum(query * keys, dim=1, keepdim=True)
    elif len(keys.shape) == 3 and mode=='paired':
        query = query.unsqueeze(1)
        logits = query @ transpose(keys)
        logits = logits.squeeze(1)
    elif len(keys.shape) == 2 and mode=='unpaired':
        logits = query @ transpose(keys)
    else:
        raise ValueError("Invalid mode")
    return logits



class WassersteinLossStab(Function):
    """Adapted from T. Viehmann, Batch Sinkhorn Iteration Wasserstein Distance, PyTorch code and notebook, 2017,
    https://github.com/t-vi/pytorch-tvmisc/blob/ae4d94597751f98d4a0d7b10188dd02c13a0c6fd/wasserstein-distance/Pytorch_Wasserstein.ipynb
    """
    def __init__(self,cost, lam = 1e-3, sinkhorn_iter = 50):
        super(WassersteinLossStab,self).__init__()

        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost/self.lam)
        self.KM = self.cost*self.K
        self.stored_grad = None

    def forward(self, pred, target):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        assert pred.size(1)==self.na
        assert target.size(1)==self.nb

        batch_size = pred.size(0)

        log_a, log_b = torch.log(pred), torch.log(target)
        log_u = self.cost.new(batch_size, self.na).fill_(-numpy.log(self.na))
        log_v = self.cost.new(batch_size, self.nb).fill_(-numpy.log(self.nb))

        for i in range(self.sinkhorn_iter):
            log_u_max = torch.max(log_u, dim=1)[0]
            u_stab = torch.exp(log_u-log_u_max.expand_as(log_u))
            log_v = log_b - torch.log(torch.mm(self.K.t(),u_stab.t()).t()) - log_u_max.expand_as(log_v)
            log_v_max = torch.max(log_v, dim=1)[0]
            v_stab = torch.exp(log_v-log_v_max.expand_as(log_v))
            log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max.expand_as(log_u)

        log_v_max = torch.max(log_v, dim=1)[0]
        v_stab = torch.exp(log_v-log_v_max.expand_as(log_v))
        logcostpart1 = torch.log(torch.mm(self.KM,v_stab.t()).t())+log_v_max.expand_as(log_u)
        wnorm = torch.exp(log_u+logcostpart1).mean(0).sum() # sum(1) for per item pair loss...
        grad = log_u*self.lam
        grad = grad-torch.mean(grad,dim=1).expand_as(grad)
        grad = grad-torch.mean(grad,dim=1).expand_as(grad) # does this help over only once?
        grad = grad/batch_size

        self.stored_grad = grad

        return self.cost.new((wnorm,))
    def backward(self, grad_output):
        #print (grad_output.size(), self.stored_grad.size())
        #print (self.stored_grad, grad_output)
        res = grad_output.new()
        res.resize_as_(self.stored_grad).copy_(self.stored_grad)
        if grad_output[0] != 1:
            res.mul_(grad_output[0])
        return res,None