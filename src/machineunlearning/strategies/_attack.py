"""Adversarial attack utilities used by boundary unlearning.

Paper: Boundary Unlearning — https://ieeexplore.ieee.org/abstract/document/10203289
Source: https://github.com/TY-LEE-KR/Boundary-Unlearning-Code
"""

import torch
import torch.distributions as distributions
from torch import nn
from torch.nn import functional as F


class AttackBase:
    def __init__(self, model=None, norm=False, discrete=True, device=None):
        self.model = model
        self.norm = norm
        if self.norm:
            self.mean = (0.4914, 0.4822, 0.2265)
            self.std = (0.2023, 0.1994, 0.2010)
        self.discrete = discrete
        self.device = device or torch.device("cuda:0")
        self.loss(device=self.device)

    def loss(self, custom_loss=None, device=None):
        device = device or self.device
        self.criterion = custom_loss or nn.CrossEntropyLoss()
        self.criterion.to(device)

    def perturb(self, x):
        raise NotImplementedError

    def normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]
            y[:, 1, :, :] = (y[:, 1, :, :] - self.mean[1]) / self.std[1]
            y[:, 2, :, :] = (y[:, 2, :, :] - self.mean[2]) / self.std[2]
            return y
        return x

    def inverse_normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
            y[:, 1, :, :] = y[:, 1, :, :] * self.std[1] + self.mean[1]
            y[:, 2, :, :] = y[:, 2, :, :] * self.std[2] + self.mean[2]
            return y
        return x

    def discretize(self, x):
        return torch.round(x * 255) / 255

    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)


class FGSM(AttackBase):
    def __init__(self, model=None, bound=None, norm=False, random_start=False, discrete=True, device=None, **kwargs):
        super().__init__(model, norm, discrete, device)
        self.bound = bound
        self.rand = random_start

    def perturb(self, x, y, model=None, bound=None, device=None, **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        device = device or self.device

        model.zero_grad()
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(
                self.inverse_normalize(x_adv) + rand_perturb,
                x_nat,
                bound=bound,
                inverse_normalized=True,
            )
            if self.discretize:
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        pred = model(x_adv)
        if criterion.__class__.__name__ == "NLLLoss":
            pred = F.softmax(pred, dim=-1)
        loss = criterion(pred, y)
        loss.backward()

        grad_sign = x_adv.grad.data.detach().sign()
        x_adv = self.inverse_normalize(x_adv) + grad_sign * bound
        x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)

        return x_adv.detach()


def inf_generator(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def _find_z(model, inputs, targets, h):
    """Finding the direction in the regularizer."""
    inputs.requires_grad_()
    outputs = model(inputs)  # noqa: F841
    loss_z = nn.CrossEntropyLoss()(model(inputs), targets)
    loss_z.backward()
    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.0
    z = 1.0 * h * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)
    inputs.grad.zero_()
    model.zero_grad()

    return z, norm_grad


def curvature(model, inputs, targets, h=3.0, lambda_=4):
    """Regularizer term in CURE."""
    z, norm_grad = _find_z(model, inputs, targets, h)

    inputs.requires_grad_()
    outputs_pos = model(inputs + z)
    outputs_orig = model(inputs)

    loss_pos = nn.CrossEntropyLoss()(outputs_pos, targets)
    loss_orig = nn.CrossEntropyLoss()(outputs_orig, targets)
    grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs, create_graph=True)[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    model.zero_grad()

    return torch.sum(lambda_ * reg) / float(inputs.size(0)), reg


def PM(logit, target):
    if logit.shape[1] == 10:
        eye = torch.eye(10).cuda()
    else:
        eye = torch.eye(11).cuda()
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
    top2_probs = logit.softmax(1).topk(2, largest=True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1, 1)).float().sum(1) == 1
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:, 0]).detach()
    return probs_2nd - probs_GT


def weight_assign(logit, target, bias, slope):
    pm = PM(logit, target)
    reweight = ((pm + bias) * slope).sigmoid().detach()
    normalized_reweight = reweight * 3
    return normalized_reweight
