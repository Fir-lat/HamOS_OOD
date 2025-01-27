import faiss.contrib.torch_utils
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing

from collections import defaultdict
from torch.autograd import grad
from scipy.special import iv


class HamOSTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Config) -> None:
        # a bunch of constants or hyperparams
        self.n_cls = config.dataset.num_classes
        self.sample_number = config.trainer.trainer_args.sample_number
        self.feat_dim = config.network.feat_dim
        try:
            self.penultimate_dim = net.backbone.feature_size
        except AttributeError:
            self.penultimate_dim = net.backbone.module.feature_size
        self.start_epoch = config.trainer.trainer_args.start_epoch
        self.K = config.trainer.trainer_args.K
        self.select = config.trainer.trainer_args.select
        self.w_disp = config.trainer.trainer_args.w_disp
        self.w_comp = config.trainer.trainer_args.w_comp
        self.loss_weight = config.trainer.trainer_args.loss_weight
        self.temp = config.trainer.trainer_args.temp

        self.bandwidth = config.trainer.trainer_args.bandwidth
        self.leapfrog = config.trainer.trainer_args.leapfrog
        self.step_size = config.trainer.trainer_args.step_size
        self.margin = config.trainer.trainer_args.step_size
        self.num_neighbor = min(config.trainer.trainer_args.num_neighbor, self.n_cls - 1) 
        self.synthesis_every = config.trainer.trainer_args.synthesis_every


        self.net = net
        self.train_loader = train_loader
        self.config = config

        num_resources = min(10, self.n_cls)
        self.res = [faiss.StandardGpuResources() for _ in range(num_resources)]
        self.kde = [KernelDensityEstimator(feat_dim=self.feat_dim, bandwidth=self.bandwidth, K=self.K, res=self.res[i % num_resources]) for i in range(self.n_cls)]

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        if config.dataset.train.batch_size * config.num_gpus * config.num_machines > 256:
            config.optimizer.warm = True

        if config.optimizer.warm:
            self.warmup_from = 0.001
            self.warm_epochs = 10
            if config.optimizer.cosine:
                eta_min = config.optimizer.lr * (config.optimizer.lr_decay_rate**3)
                self.warmup_to = eta_min + (config.optimizer.lr - eta_min) * (1 + math.cos(math.pi * self.warm_epochs / config.optimizer.num_epochs)) / 2.0
            else:
                self.warmup_to = config.optimizer.lr

        self.criterion_comp = CompLoss(self.n_cls, temperature=self.temp).cuda()
        self.criterion_comp_ood = CompLoss_ood(self.n_cls, temperature=self.temp).cuda()
        # V2: EMA style prototypes
        self.criterion_disp = DispLoss(
            self.n_cls,
            self.feat_dim,
            config.trainer.trainer_args.proto_m,
            self.net,
            val_loader,
            temperature=self.temp
        ).cuda()

        num_pairs = (self.n_cls * (self.n_cls - 1)) // 2
        # Initialize the mask tensor
        self.mask = torch.zeros((self.n_cls, num_pairs), requires_grad=True).int().cuda()
        # Create a meshgrid of indices
        indices_i, indices_j = torch.triu_indices(self.n_cls, self.n_cls, offset=1)
        # Set the mask values
        self.mask[indices_i, torch.arange(num_pairs)] = 1
        self.mask[indices_j, torch.arange(num_pairs)] = 1

        self.data_dict = torch.zeros(self.n_cls, self.sample_number, self.feat_dim).cuda()

        self.number_dict = [0] * self.n_cls

    def train_epoch(self, epoch_idx):
        adjust_learning_rate(self.config, self.optimizer, epoch_idx - 1)

        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc='Epoch {:03d}: '.format(epoch_idx),
            position=0,
            leave=True,
            disable=not comm.is_main_process()
        ):
            if self.config.optimizer.warm:
                warmup_learning_rate(self.config, self.warm_epochs,
                                 self.warmup_from,
                                 self.warmup_to, epoch_idx - 1, train_step,
                                 len(train_dataiter), self.optimizer)

            batch = next(train_dataiter)
            data = batch['data']
            target = batch['label']

            data = torch.cat([data[0], data[1]], dim=0).cuda()
            target = target.repeat(2).cuda()

            # forward
            logits, feature = self.net.forward(data, return_feature=True)


            # cache ID features
            sum_temp = sum(self.number_dict)
            if sum_temp == self.n_cls * self.sample_number:
                target_numpy = target.cpu().data.numpy()
                for idx in range(len(target_numpy)):
                    dict_key = target[idx]
                    self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:], feature[idx].detach().view(1, -1)), 0)
            else:
                target_numpy = target.cpu().data.numpy()
                for idx in range(len(target_numpy)):
                    dict_key = target[idx]
                    if self.number_dict[dict_key] < self.sample_number:
                        self.data_dict[dict_key][self.number_dict[dict_key]] = feature[idx].detach()
                        self.number_dict[dict_key] += 1

            criterion_CE = nn.CrossEntropyLoss()
            ce_loss = criterion_CE(logits, target)
            disp_loss = self.criterion_disp(feature, target)
            comp_loss = self.criterion_comp(feature, self.criterion_disp.prototypes, target)
            loss = self.w_disp * disp_loss + self.w_comp * comp_loss + ce_loss
        
            
            lr_reg_loss = torch.zeros(1).cuda()[0]
            if sum_temp == self.n_cls * self.sample_number and epoch_idx >= self.start_epoch and train_step % self.synthesis_every == 0:
                for cls_idx in range(self.n_cls):
                    self.kde[cls_idx].fit(self.data_dict[cls_idx])

                means = torch.stack([torch.mean(self.data_dict[i], dim=0) for i in range(self.n_cls)])
                centroids, mask = initialize_states(means, n_cls=self.n_cls, num_neighbor=self.num_neighbor)
                ood_samples = generate_outliers(
                    init_points=centroids, 
                    num_samples=self.select,
                    feat_dim=self.feat_dim, 
                    kdes=self.kde,
                    mask=mask,
                    num_steps=self.leapfrog,
                    step_size=self.step_size,
                    margin = self.margin
                )

                if len(ood_samples) != 0:
                    lr_reg_loss = self.criterion_comp_ood(feature, self.criterion_disp.prototypes)
                    

            loss = loss + self.loss_weight * lr_reg_loss # ce_loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2


        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced


def adjust_learning_rate(config, optimizer, epoch):
    lr = config.optimizer.lr
    if config.optimizer.cosine:
        eta_min = lr * (config.optimizer.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / config.optimizer.num_epochs)) / 2.0
    else:
        steps = np.sum(epoch > np.asarray(config.optimizer.lr_decay_epochs))
        if steps > 0:
            lr = lr * (config.optimizer.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(config, warm_epochs, warmup_from, warmup_to, epoch, batch_id, total_batches, optimizer):
    if config.optimizer.warm and epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class CompLoss_ood(nn.Module):
    def __init__(self, n_cls, temperature=0.07, base_temperature=0.07):
        super(CompLoss_ood, self).__init__()
        self.n_cls = n_cls
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes):

        # compute logits
        contrast_feature = prototypes.clone()
        anchor_dot_contrast = torch.div(
            torch.matmul(features, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_min.detach()

        # compute log_prob
        exp_logits = torch.exp(-logits)
        log_prob = logits + torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = torch.mean(log_prob, dim=1)
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss


class CompLoss(nn.Module):
    def __init__(self, n_cls, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.n_cls = n_cls
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        device = torch.device('cuda')

        proxy_labels = torch.arange(0, self.n_cls).to(device)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, proxy_labels.T).float().to(device)

        # compute logits
        anchor_feature = features
        contrast_feature = prototypes / prototypes.norm(dim=-1, keepdim=True)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss


class DispLoss(nn.Module):
    def __init__(self,
                 n_cls,
                 feat_dim,
                 proto_m,
                 model,
                 loader,
                 temperature=0.1,
                 base_temperature=0.1):
        super(DispLoss, self).__init__()
        self.n_cls = n_cls
        self.feat_dim = feat_dim
        self.proto_m = proto_m
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer('prototypes',
                             torch.zeros(self.n_cls, self.feat_dim))
        self.model = model
        self.loader = loader
        self.init_class_prototypes()

    def forward(self, features, labels):
        prototypes = self.prototypes
        num_cls = self.n_cls
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(
                prototypes[labels[j].item()] * self.proto_m + features[j] *
                (1 - self.proto_m),
                dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)

        mask = (1 - torch.eq(labels, labels.mT).float()).cuda()

        logits = torch.div(torch.matmul(prototypes, prototypes.mT),
                           self.temperature)

        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(num_cls).view(-1, 1).cuda(),
                                    0)
        mask = mask * logits_mask
        mean_prob_neg = torch.log(
            (mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self):
        """Initialize class prototypes."""
        self.model.eval()
        start = time.time()
        prototype_counts = [0] * self.n_cls
        with torch.no_grad():
            prototypes = torch.zeros(self.n_cls, self.feat_dim).cuda()
            for i, batch in enumerate(self.loader):
                input = batch['data']
                target = batch['label']
                input, target = input.cuda(), target.cuda()
                logits, features = self.model.forward(input, return_feature=True)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.n_cls):
                prototypes[cls] /= prototype_counts[cls]
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes

def initialize_states(centroids, num_neighbor=5, n_cls=10):
    cos_sim = torch.mm(centroids, centroids.T)
    cos_sim.fill_diagonal_(-1)
    _, near_indices = torch.topk(cos_sim, num_neighbor, dim=1, largest=True)

    pairs = torch.zeros(n_cls * num_neighbor, 2, dtype=torch.long)
    for i in range(n_cls):
        pairs[i * num_neighbor: (i + 1) * num_neighbor, 0] = i
        pairs[i * num_neighbor: (i + 1) * num_neighbor, 1] = near_indices[i]

    midpoints = (centroids[pairs[:, 0]] + centroids[pairs[:, 1]]) * 0.5
    midpoints = F.normalize(midpoints, dim=1)

    mask =  torch.zeros((n_cls, n_cls * num_neighbor), requires_grad=True).int().cuda()
    mask[pairs[:, 0], torch.arange(n_cls * num_neighbor)] = 1
    mask[pairs[:, 1], torch.arange(n_cls * num_neighbor)] = 1

    return midpoints, mask

class KernelDensityEstimator:
    def __init__(self, kernel='vmf', feat_dim=512, bandwidth=3.0, K=200, res=None):
        self.kernel = kernel
        self.bandwidth = bandwidth  # This will be the concentration parameter kappa for vMF
        self.feat_dim = feat_dim # feature dimension
        self.K = K # k value for nearest neighbors using faiss index
        self.res = res
        self.index = None # faiss index
        self.data = None # training data
        self.bessel = (2.0 * math.pi) ** (feat_dim * 0.5) * iv(feat_dim * 0.5 - 1, bandwidth)

    def fit(self, data):
        self.data = data
        if self.index is None:
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = True
            cfg.device = list(range(torch.cuda.device_count()))[0]
            # self.res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(self.res, self.feat_dim, cfg)
        else:
            self.index.reset()
        self.index.add(data)

    def vmf_kernel(self, dot_products):
        log_kernel = self.bandwidth * dot_products - math.log(self.bessel)
        log_max, _ = torch.max(log_kernel, dim=1, keepdim=True)
        kernel_values = torch.exp(log_kernel - log_max)
        return kernel_values

    def score_samples(self, points):
        with torch.no_grad():
            distances, indices = self.index.search(points, self.K)  
        
        retrieved_data = self.data[indices]
        far_points = retrieved_data[:, -1]
        l2_dist = torch.norm(points - far_points, p=2, dim=1)

        density = l2_dist #(torch.sum(kernel_values, dim=1) / self.K) * l2_dist
        return density

    def pdf(self, points):
        with torch.no_grad():
            distances, indices = self.index.search(points, self.K)  

        retrieved_data = self.data[indices]
        dot_products = torch.bmm(points.unsqueeze(1), retrieved_data.transpose(1, 2)).squeeze(1)

        if self.kernel == 'vmf':
            kernel_values = self.vmf_kernel(dot_products)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        prob = torch.sum(kernel_values, dim=1) / self.K

        return prob


class SphericalHMC(nn.Module):
    def __init__(self, id_pdf, logpdf, dim, L, eps, margin=0.5):
        super(SphericalHMC, self).__init__()
        self.id_pdf = id_pdf
        self.logpdf = logpdf
        self.dim = dim
        self.L = L
        self.eps = eps
        self.margin = margin
        self.I_a = nn.parameter.Parameter(torch.eye(self.dim), requires_grad=False).cuda()
        self.threshold_min = None

    @staticmethod
    def batch_square_norm(input):
        return torch.sum(torch.pow(input, 2), -1)

    @staticmethod
    def batch_outer(input1, input2):
        return torch.einsum('bi,bj->bij', input1, input2)

    @staticmethod
    def batch_mvp(input1, input2):
        return torch.einsum('bij,bj->bi', input1, input2)

    def H(self, theta_a, v):
        u = - self.logpdf(theta_a)
        return u + self.batch_square_norm(v) / 2.

    def sample(self, theta_0):
        if self.threshold_min is None:
            log_pdf_val = self.id_pdf(theta_0)
            self.threshold_min = - log_pdf_val - (self.margin * 0.5)

        theta_a = theta_0
        v = torch.randn_like(theta_a)
        v -= SphericalHMC.batch_mvp(SphericalHMC.batch_outer(theta_a, theta_a), v)
        h_0 = self.H(theta_a, v)
        for _ in range(self.L):
            theta = theta_a.detach().clone()
            theta.requires_grad = True
            u = -self.logpdf(theta)
            g = grad(u, theta, torch.ones_like(u))[0]
            theta.requires_grad = False
            v -= self.eps / 2. * SphericalHMC.batch_mvp(
                self.I_a - SphericalHMC.batch_outer(theta_a, theta), g)
            v_norm = torch.unsqueeze(torch.norm(v, dim=-1), -1)
            theta_a_new = theta_a * torch.cos(v_norm * self.eps) + \
                v / v_norm * torch.sin(v_norm * self.eps)
            v = -theta_a * v_norm * torch.sin(v_norm * self.eps) + \
                v * torch.cos(v_norm * self.eps)
            theta_a = theta_a_new
            theta = theta_a.detach().clone()
            theta.requires_grad = True
            u = -self.logpdf(theta)
            g = grad(u, theta, torch.ones_like(u))[0]
            theta.requires_grad = False
            v -= self.eps / 2. * SphericalHMC.batch_mvp(
                self.I_a - SphericalHMC.batch_outer(theta_a, theta), g)
        h = self.H(theta_a, v)
        delta_H = h - h_0
        id_pdfs = -self.id_pdf(theta)
        mask = torch.unsqueeze(torch.logical_and(
            torch.rand_like(delta_H) < torch.exp(-delta_H), 
            id_pdfs > self.threshold_min
        ), -1)
        return theta * mask + theta_0 * ~mask


def generate_outliers(init_points, num_samples, feat_dim, kdes, mask, num_steps=3, step_size=0.1, margin=0.5):
    def logprob(x):
        result = torch.zeros(len(x), requires_grad=True).float().cuda()
        for i in range(len(kdes)):
            selected_points = x[mask[i].bool()]
            probs = kdes[i].score_samples(selected_points)
            result.index_add_(0, mask[i].nonzero().squeeze(), probs)
        result = result * 0.5
        return result.log()

    def id_pdf(x):
        pdfs = torch.stack([kde.pdf(x) for kde in kdes], dim=1)
        softmax_pdf = F.softmax(pdfs, dim=1)
        max_pdf = torch.max(softmax_pdf, dim=1).values
        return max_pdf.log()

    shmc = SphericalHMC(id_pdf, logprob, feat_dim, num_steps, step_size, margin)

    samples = [init_points]
    cur_positions = init_points
    cnt = 0
    for _ in range(num_samples):
        new_positions = shmc.sample(cur_positions)
        if not torch.isnan(new_positions).any():
            cnt += 1
            cur_positions = new_positions
        samples.append(cur_positions)

    result = torch.cat(samples, dim=0)
    return result


