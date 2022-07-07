import numpy as np
from numpy import random
import numpy as np
import torch
from torch.utils.data import Subset
import torch.nn.functional as F

# np.random.seed(1)

def label_to_onehot(target, num_classes):

    target = torch.unsqueeze(target, 1) #[9,9]->[[9],[9]]
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

class Loss:
    """Abstract class, containing necessary methods.
    Abstract class to collect information about the 'higher-level' loss function, used to train an energy-based model
    containing the evaluation of the loss function, its gradients w.r.t. to first and second argument and evaluations
    of the actual metric that is targeted. """

    def __init__(self):
        """Init."""
        pass

    def __call__(self, reference, argmin):
        """Return l(x, y)."""
        raise NotImplementedError()
        return value, name, format

    def metric(self, reference, argmin):
        """The actually sought metric."""
        raise NotImplementedError()
        return value, name, format


class PSNR(Loss):
    """A classical MSE target.
    The minimized criterion is MSE Loss, the actual metric is average PSNR.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'MSE'
        format = '.6f'
        if x is None:
            return name, format
        else:
            value = 0.5 * self.loss_fn(x, y)
            return value, name, format

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'avg PSNR'
        format = '.3f'
        if x is None:
            return name, format
        else:
            value = self.psnr_compute(x, y)
            return value, name, format

    @staticmethod
    def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0):
        """Standard PSNR."""
        def get_psnr(img_in, img_ref):
            mse = ((img_in - img_ref)**2).mean()
            if mse > 0 and torch.isfinite(mse):
                return (10 * torch.log10(factor**2 / mse)).item()
            elif not torch.isfinite(mse):
                return float('nan')
            else:
                return float('inf')

        if batched:
            psnr = get_psnr(img_batch.detach(), ref_batch)
        else:
            [B, C, m, n] = img_batch.shape
            psnrs = []
            for sample in range(B):
                psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
            psnr = np.mean(psnrs)

        return psnr


class Classification(Loss):
    """A classical NLL loss for classification. Evaluation has the softmax baked in.
    The minimized criterion is cross entropy, the actual metric is total accuracy.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                 reduce=None, reduction='mean')

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'CrossEntropy'
        format = '1.5f'
        if x is None:
            return name, format
        else:
            value = self.loss_fn(x, y)
            return value, name, format

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'Accuracy'
        format = '6.2%'
        if x is None:
            return name, format
        else:
            value = (x.data.argmax(dim=1) == y).sum().float() / y.shape[0]
            return value.detach(), name, format

def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()

def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
            
def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def TVloss_l1(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    # count_h = _tensor_size(x[:, :, 1:, :])
    # count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])).sum()
    w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1])).sum()
    return torch.sum(h_tv + w_tv) / batch_size

def TVloss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def MSE(A, B):
    MSE = 0
    index = []
    for j in range(A.shape[0]):
        a = []
        for i in range(A.shape[0]):
            a.append(np.mean((A[j] - B[i]) ** 2))
        MSE += np.mean((A[j] - B[np.argsort(a)[0]]) ** 2)
        index.append(np.argsort(a)[0])

    return MSE / A.shape[0]


