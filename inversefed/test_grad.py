"""Mechanisms for image reconstruction from parameter gradients."""

import torch
import torch.nn as nn
from collections import defaultdict, OrderedDict
from .metrics import total_variation as TV
from .metrics import InceptionScore
from .medianfilt import MedianPool2d
from copy import deepcopy
from functools import partial
import time



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


def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target




class GradientReconstructor_test():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=dict(), num_images=1):
        """Initialize with algorithm setup."""
        self.config = config
        self.model = model
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.setup = dict(device=self.device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        #BN Statistics
        self.bn_layers = []
        # if self.config['bn_stat'] > 0:
        #     for module in model.modules():
        #         if isinstance(module, nn.BatchNorm2d):
        #             self.bn_layers.append(BNStatisticsHook(module))
        self.bn_prior = None
        
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
            for trial in range(self.config['restarts']):

                print("______________________________")
                
                #x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                x_trial = x[trial]
                 
                x_trial.requires_grad = True
                if self.reconstruct_label:
                    output_test = self.model(x_trial)
                    labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

                    if self.config['optim'] == 'adam':
                        optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
                    elif self.config['optim'] == 'sgd':  # actually gd
                        optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
                    elif self.config['optim'] == 'LBFGS':
                        optimizer = torch.optim.LBFGS([x_trial, labels])
                    else:
                        raise ValueError()
                else:
                    if self.config['optim'] == 'adam':
                        optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
                    elif self.config['optim'] == 'sgd':  # actually gd
                        optimizer = torch.optim.SGD([x_trial], lr=0.01) # momentum=0.9,  nesterov=True,
                    elif self.config['optim'] == 'LBFGS':
                        optimizer = torch.optim.LBFGS([x_trial])
                    else:
                        raise ValueError()

                max_iterations = self.config['max_iterations']
                dm, ds = self.mean_std
                if self.config['lr_decay']:
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                    milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                                max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
                try:
                    for iteration in range(max_iterations):
                    
                        closure = self.gradient_closure(optimizer, x_trial, input_data, labels)
                        
                        rec_loss = optimizer.step(closure)

                        #print(optimizer.state_dict()['param_groups'])    
                        
                        if self.config['lr_decay']:
                            scheduler.step()

                        with torch.no_grad():
                            # Project into image space
                            if self.config['boxed']:
                                x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)
                            if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                                print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                            if (iteration + 1) % 500 == 0:
                                if self.config['filter'] == 'none':
                                    pass
                                elif self.config['filter'] == 'median':
                                    print("addition")
                                    x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                                else:
                                    raise ValueError()

                        if dryrun:
                            break
                except KeyboardInterrupt:
                    print(f'Recovery interrupted manually in iteration {iteration}!')
                    pass
                

                # Finalize
                return x_trial.detach(), None

                
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
  

        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # # Choose optimal result:
        # if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
        #     x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        # else:
        #     print('Choosing optimal result ...')
        #     scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
        #     optimal_index = torch.argmin(scores)
        #     print(f'Optimal result score: {scores[optimal_index]:2.4f}')
        #     stats['opt'] = scores[optimal_index].item()
        #     x_optimal = x[optimal_index]

        # print(f'Total time: {time.time()-start_time}.')
        # return x_optimal.detach(), stats

    def _init_images(self, img_shape):

        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)

        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()


    def gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():
            
            optimizer.zero_grad()
            self.model.zero_grad()
        
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss =  reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            # if self.config['total_variation'] > 0:
            #     tv_loss = TV(x_trial)
            #     rec_loss += self.config['total_variation'] * tv_loss
            
            rec_loss.backward()
            x_trial.grad.sign_()
    
            return rec_loss
        return closure



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

        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)

        

