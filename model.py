import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time

from burgers import *

import matplotlib.pyplot as plt

from copy import copy

class PINN(nn.Module):
    
    def __init__(self, neuron_num, layer_num, zero_shot=False, dim=1, param_num=2):

        """Initializes Physics-Informed Neural Networks.
         Model architecture from Raissi et al. (2019)
         The model is MLP, consisting of layer_num fully-connected layers
         with neuron_num neurons per layer. Each output vector from a layer
         is activated by tanh function.

        Args:
            neuron_num (int): the number of neurons in a single layer
            layer_num (int): the number of layers
            zero_shot (boolean): indicating zero-shot learning or not
            dim (int): dimension of the problem (e.g. 1 for 1-D, 2 for 2-D)
            param_num (int): the number of parameters w.r.t. zero-shot learning (e.g. 2 for alpha, beta)
        """

        super(PINN, self).__init__()

        layers = []

        for i in range(layer_num):
            if i == 0:
                layer = nn.Linear(dim + param_num, neuron_num) if zero_shot else nn.Linear(dim, neuron_num)
            elif i == layer_num - 1:
                layer = nn.Linear(neuron_num, 1)
            else:
                layer = nn.Linear(neuron_num, neuron_num)

            layers.append(layer)

        self.module1 = nn.Sequential(*layers)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.zero_shot = zero_shot
    
    def forward(self, input):
        # input      = torch.cat([x, t], axis=1)
        act_func        = nn.Tanh()
        
        tmp = input
        for n, layer in enumerate(self.module1):
            if n == len(self.module1) - 1:
                tmp = layer(tmp)
                break
            tmp = act_func(layer(tmp))
        
        return tmp

    def __call__(self, input):
        return self.forward(input)

    
    def calc_loss_f(self, input, target, alpha=None, beta=None):
        u_hat = self(input)
        # print(input)
        if alpha is None:
            alpha = input[0][-2]
            beta = input[0][-1]
        x = input[:, 0].reshape(-1, 1)

        
        deriv_1 = autograd.grad(u_hat.sum(), input, create_graph=True)
        u_hat_x = deriv_1[0][:, 0].reshape(-1, 1)
        # u_hat_t = deriv_1[0][:, 1].reshape(-1, 1)
        deriv_2 = autograd.grad(u_hat_x.sum(), input, create_graph=True)
        u_hat_x_x = deriv_2[0][:, 0].reshape(-1, 1)

        # to modify governing equation, modify here
        # f = u_hat_t + u_hat * u_hat_x - z * u_hat_x_x
        f = u_hat_x_x + alpha ** 2 * torch.sin(alpha * x) + beta ** 2 * torch.cos(alpha * x)
        # f = u_hat_t - u_hat_x_x
        func = nn.MSELoss()
        return func(f, target)

    def calc_loss_f_burgers(self, input, target, alpha=None):
        u_hat = self(input)
        if alpha is None:
            alpha = input[0][-1]
        x = input[:, 0].reshape(-1, 1)

        # print(alpha)
        deriv_1 = autograd.grad(u_hat.sum(), input, create_graph=True)
        u_hat_x = deriv_1[0][:, 0].reshape(-1, 1)
        u_hat_t = deriv_1[0][:, 1].reshape(-1, 1)
        deriv_2 = autograd.grad(u_hat_x.sum(), input, create_graph=True)
        u_hat_x_x = deriv_2[0][:, 0].reshape(-1, 1)

        # to modify governing equation, modify here
        f = u_hat_t + u_hat * u_hat_x - alpha * u_hat_x_x
        # f = u_hat_x_x + alpha ** 2 * torch.sin(alpha * x) + beta ** 2 * torch.cos(alpha * x)
        # f = u_hat_t - u_hat_x_x
        func = nn.MSELoss()
        return func(f, target)
    
    def validate(self, alpha, beta, low=-1, high=1):
        """Validate the model via comparison between prediction and ground truth (mostly analytical solution).

        args:
            alpha (float): alpha of the given equation
            beta (float): beta of the given equation

        Returns:
            nrmse (float): NRMSE between prediction and ground truth

        """

        x = np.linspace(low, high, num=100).reshape(-1, 1)
        X = torch.tensor(x, dtype=torch.float32).to(self.device)
        alpha_full = np.full((x.shape), alpha)
        beta_full = np.full((x.shape), beta)
        pred = self(torch.Tensor(np.hstack((x, alpha_full, beta_full))).to(self.device)).detach().cpu().numpy() if self.zero_shot else self(X).detach().cpu().numpy()
        sol  = np.sin(alpha * x) + np.cos(beta * x) + 0.1 * x
        nrmse = np.sqrt(np.sum((pred - sol) ** 2) / np.sum(pred ** 2))
        # plt.cla()
        # plt.scatter(x, sol, label='answer')
        # plt.scatter(x, pred, label='pred')
        # plt.legend()
        # plt.savefig('test_{:.3f}.png'.format(nrmse))

        return nrmse

    def get_burgers(self, alpha, vtn=101, vxn=101):
        vx = np.linspace(-1, 1, vxn)
        vt = np.linspace(0, 1, vtn)
        truth = burgers_viscous_time_exact1(alpha, vxn, vx, vtn, vt).T.reshape(-1, 1)
        return truth

    def validate_burgers(self, model, alpha, truth, x_low=-1, x_high=1, t_low=0, t_high=1):
        """Validate the model via comparison between prediction and ground truth (mostly analytical solution).

        args:
            alpha (float): alpha of the given equation
            beta (float): beta of the given equation

        Returns:
            nrmse (float): NRMSE between prediction and ground truth

        """

        vtn = 101
        vxn = 101
        vx = np.linspace(-1, 1, vxn)
        vt = np.linspace(0, 1, vtn)
        x, t = np.meshgrid(vx, vt)
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)
        nu = np.full(x.shape, alpha)
        pred = model(torch.Tensor(np.hstack((x, t, nu))).to(self.device)).detach().cpu().numpy() if self.zero_shot else model(torch.Tensor(np.hstack((x, t))).to(self.device)).detach().cpu().numpy()
        nrmse = np.sqrt( np.sum( (pred - truth) ** 2) / np.sum(pred ** 2))
        # plt.cla()
        # plt.scatter(x,t,c=pred)
        # plt.savefig("pred{:.3f}.png".format(nrmse))
        # plt.cla()
        # plt.scatter(x,t,c=truth)
        # plt.savefig("truth{:.3f}.png".format(nrmse))
        return nrmse

if __name__ == "__main__":
    a = PINN(5, 5)
    