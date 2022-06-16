import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import torch

from burgers import * 

def calc_deriv(x, input, times):
    if times == 0:
        return input
    res = input
    for _ in range(times):
        res = autograd.grad(res.sum(), x, create_graph=True)[0]
    return res

def sample_data(size, lb, rb):
    return [np.random.rand() * (rb - lb) + lb for _ in range(size)]

def ground_truth(x, alpha, beta):
    return np.sin(alpha * x) + np.cos(beta * x) + 0.1 * x

def evaluate(x, alpha, beta, model, device):
    loss_func_val = nn.MSELoss(reduction='sum')
    loss_val = 0
    for a, b in zip(alpha, beta):
        input_alpha = np.full((len(x), 1), a)
        input_beta = np.full((len(x), 1), b)
        input_val = np.hstack((x, input_alpha, input_beta))
        output_val = model(torch.Tensor(input_val).to(device)).detach().cpu()
        truth_val = torch.Tensor(ground_truth(x, a, b))
        loss_val += torch.sqrt(loss_func_val(output_val, truth_val) / torch.sum(output_val ** 2))
    
    loss_val /= len(alpha)

    return loss_val

def evaluate_burgers(x, t, alpha, model, device, zero_shot):
    # loss_func_val = nn.MSELoss(reduction='sum')
    loss_val = 0
    vtn = 101
    vxn = 101
    vx = np.linspace(-1, 1, vxn)
    vt = np.linspace(0, 1, vtn)
    x, t = np.meshgrid(vx, vt)
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)
    for a in alpha:
        nu = np.full((x.shape), a)
        pred = model(torch.Tensor(np.hstack((x, t, nu))).to(device)).detach().cpu().numpy() if zero_shot else model(torch.Tensor(np.hstack((x, t))).to(device)).detach().cpu().numpy()
        truth = burgers_viscous_time_exact1(a, vxn, vx, vtn, vt).T.reshape(-1, 1)
        loss_val += np.sqrt(np.sum( (pred - truth) ** 2) / np.sum(pred ** 2))

    return loss_val