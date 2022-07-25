"""Mechanisms for image reconstruction from parameter gradients."""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict, OrderedDict
from .metrics import total_variation as TV
from copy import deepcopy


import time

imsize_dict = {
    'ImageNet': 224, 'I128':128, 'I64': 64, 'I32':32,
    'CIFAR10':32, 'CIFAR100':32, 'FFHQ':512,
    'CA256': 256, 'CA128': 128, 'CA64': 64, 'CA32': 32, 
    'PERM64': 64, 'PERM32': 32,
}

save_interval=100
construct_group_mean_at = 500
construct_gm_every = 100
DEFAULT_CONFIG = dict(signed=False,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      epochs=4800,
                      total_variation=1e-1,
                      bn_stat=1e-1,
                      image_norm=1e-1,
                      group_lazy=1e-1,
                      init='randn',
                      lr_decay=True,
                      dataset='CIFAR10',
                      gen_dataset='',
                      )

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class BNStatisticsHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        # r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
        #     module.running_mean.data - mean, 2)
        mean_var = [mean, var]

        self.mean_var = mean_var
        # must have no output

    def close(self):
        self.hook.remove()


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, bn_prior=((0.0, 1.0))):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.setup = dict(device=self.device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        #BN Statistics
        self.bn_layers = []
        if self.config['bn_stat'] > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn_layers.append(BNStatisticsHook(module))
        self.bn_prior = bn_prior
        
        #Group Regularizer
        self.do_group_mean = False
        self.group_mean = None
        
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()

        if torch.is_tensor(input_data[0]):
            input_data = [input_data]
        self.image_size = img_shape[1]
        
        stats = defaultdict(list)
        x = self._init_images(img_shape)
        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            # labels = [None for _ in range(self.config['restarts'])]
            
            optimizer = [None for _ in range(self.config['restarts'])]
            scheduler = [None for _ in range(self.config['restarts'])]
            _x = [None for _ in range(self.config['restarts'])]
            epochs = self.config['epochs']

            for trial in range(self.config['restarts']):
                _x[trial] = x[trial]

                _x[trial].requires_grad = True
                if self.config['optim'] == 'adam':
                    optimizer[trial] = torch.optim.Adam([_x[trial]], lr=self.config['lr'])
                elif self.config['optim'] == 'sgd':  # actually gd
                    optimizer[trial] = torch.optim.SGD([_x[trial]], lr=0.01, momentum=0.9, nesterov=True)
                elif self.config['optim'] == 'LBFGS':
                    optimizer[trial] = torch.optim.LBFGS([_x[trial]])
                else:
                    raise ValueError()

                if self.config['lr_decay']:
                    scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial],
                                                                        milestones=[epochs // 2.667, epochs // 1.6,

                                                                                    epochs // 1.142], gamma=0.1)   # 3/8 5/8 7/8
            dm, ds = self.mean_std
            
            
            for iteration in range(epochs):
                for trial in range(self.config['restarts']):
                    losses = [0,0,0]
                    # x_trial = _x[trial]
                    # x_trial.requires_grad = True
                    
                    #Group Regularizer
                    if trial == 0 and iteration + 1 == construct_group_mean_at and self.config['group_lazy'] > 0:
                        self.do_group_mean = True
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                    if self.do_group_mean and trial == 0 and (iteration + 1) % construct_gm_every == 0:
                        print("construct group mean")
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()
               
                    # print(x_trial)
                    closure = self._gradient_closure(optimizer[trial], _x[trial], input_data, labels, losses)
                    rec_loss = optimizer[trial].step(closure)
                    if self.config['lr_decay']:
                        scheduler[trial].step()

                    with torch.no_grad():
                        # Project into image space
                        _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                        if (iteration + 1 == epochs) or iteration % save_interval == 0:
                            print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | l2: {losses[2]:7.4f} | gr: {losses[3]:7.4f}')

                    if dryrun:
                        break

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        

        for trial in range(self.config['restarts']):
            x[trial] = _x[trial].detach()
            scores[trial] = self._score_trial(x[trial], input_data, labels)
            if tol is not None and scores[trial] <= tol:
                break
            if dryrun:
                break

        # Choose optimal result:
        print('Choosing optimal result ...')
        scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
        optimal_index = torch.argmin(scores)
        print(f'Optimal result score: {scores[optimal_index]:2.4f}')
        stats['opt'] = scores[optimal_index].item()
        x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats


    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, losses):

        def closure():
            num_images = label.shape[0]
            num_gradients = len(input_gradient)
            batch_size = num_images // num_gradients
            num_batch = num_images // batch_size

            total_loss = 0
            optimizer.zero_grad()
            self.model.zero_grad()
            for i in range(num_batch):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_input = x_trial[start_idx:end_idx]
                batch_label = label[start_idx:end_idx]
                loss = self.loss_fn(self.model(batch_input), batch_label)
                gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                                cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                                weights=self.config['weights'])

                if self.config['total_variation'] > 0:
                    tv_loss = TV(x_trial)
                    rec_loss += self.config['total_variation'] * tv_loss
                    losses[0] = tv_loss
                if self.config['bn_stat'] > 0:
                    bn_loss = 0
                    first_bn_multiplier = 10.
                    rescale = [first_bn_multiplier] + [1. for _ in range(len(self.bn_layers)-1)]
                    for i, (my, pr) in enumerate(zip(self.bn_layers, self.bn_prior)):
                        bn_loss += rescale[i] * (torch.norm(pr[0] - my.mean_var[0], 2) + torch.norm(pr[1] - my.mean_var[1], 2))
                    rec_loss += self.config['bn_stat'] * bn_loss
                    losses[1] = bn_loss
                if self.config['image_norm'] > 0:
                    norm_loss = torch.norm(x_trial, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['image_norm'] * norm_loss
                    losses[2] = norm_loss
                if self.do_group_mean and self.config['group_lazy'] > 0:
                    group_loss =  torch.norm(x_trial - self.group_mean, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['group_lazy'] * group_loss
                    losses[3] = group_loss
                total_loss += rec_loss
            total_loss.backward()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        num_images = label.shape[0]
        num_gradients = len(input_gradient)
        batch_size = num_images // num_gradients
        num_batch = num_images // batch_size

        total_loss = 0
        for i in range(num_batch):
            self.model.zero_grad()
            x_trial.grad = None

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_input = x_trial[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            loss = self.loss_fn(self.model(batch_input), batch_label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                    cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                    weights=self.config['weights'])
            total_loss += rec_loss
        return total_loss


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]

    else:
        raise ValueError()

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
            elif cost_fn == 'gaussian':
                l2 = (trial_gradient[i] - input_gradient[i]).pow(2).sum()
                sigma = torch.var(input_gradient[i]) * torch.var(input_gradient[i])
                costs += (1 - torch.exp(-l2/sigma)) * weights[i]

            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                input_gradient[i].flatten(),
                                                                0, 1e-10) * weights[i]

        if cost_fn.startswith('sim'):
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)

            # elif cost_fn.startswith('compressed'):
            #     ratio = float(cost_fn[10:])
            #     k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
            #     k = max(k, 1)

            #     trial_flatten = trial_gradient[i].flatten()
            #     trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
            #     trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
            #     trial_compressed = trial_flatten * trial_mask

            #     input_flatten = input_gradient[i].flatten()
            #     input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
            #     input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
            #     input_compressed = input_flatten * input_mask
            #     costs += ((trial_compressed - input_compressed).pow(2)).sum() * weights[i]
            # elif cost_fn.startswith('sim_cmpr'):
            #     ratio = float(cost_fn[8:])
            #     k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
            #     k = max(k, 1)
                
            #     input_flatten = input_gradient[i].flatten()
            #     input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
            #     input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
            #     input_compressed = input_flatten * input_mask

            #     trial_flatten = trial_gradient[i].flatten()
            #     # trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
            #     # trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
            #     trial_compressed = trial_flatten * input_mask

                
            #     costs -= (trial_compressed * input_compressed).sum() * weights[i]
            #     pnorm[0] += trial_compressed.pow(2).sum() * weights[i]
            #     pnorm[1] += input_compressed.pow(2).sum() * weights[i]