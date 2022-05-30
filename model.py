import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time


from copy import copy

class PINN(nn.Module):
    
    def __init__(self, neuron_num, layer_num, zero_shot=False):

        """Initializes Physics-Informed Neural Networks.
         Model architecture from Raissi et al. (2019)
         The model is MLP, consisting of layer_num fully-connected layers
         with neuron_num neurons per layer. Each output vector from a layer
         is activated by tanh function.

        Args:
            neuron_num (int): the number of neurons in a single layer
            layer_num (int): the number of layers
            zero_shot (boolean): indicating zero-shot learning or not
        """

        super(PINN, self).__init__()

        layers = []

        for i in range(layer_num):
            if i == 0:
                layer = nn.Linear(3, neuron_num) if zero_shot else nn.Linear(1, neuron_num)
            elif i == layer_num - 1:
                layer = nn.Linear(neuron_num, 1)
            else:
                layer = nn.Linear(neuron_num, neuron_num)

            layers.append(layer)

        self.module1 = nn.Sequential(*layers)
    
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
    
if __name__ == "__main__":
    a = PINN(5, 5)
    