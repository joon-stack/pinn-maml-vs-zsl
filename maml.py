import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os


from model import *
from generate_data import *
from utils import *
from copy import deepcopy

class MAML:
    def __init__(self, num_inner_steps, inner_lr, outer_lr):

        """Initializes First-Order Model-Agnostic Meta-Learning.
        Model architecture from https://arxiv.org/abs/1703.03400
        The model consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and a ReLU activation.

        Args:
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
            outer_lr (float): learning rate for outer-loop optimization
        """

        print("Initializing MAML-PINN model")

        self.model = PINN(20, 5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._save_dir = 'models/maml.data'
        os.makedirs(self._save_dir, exist_ok=True)

        self._num_inner_steps = num_inner_steps
        
        self._inner_lr = inner_lr
        self._outer_lr = outer_lr
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self._outer_lr)

        print("Finished initialization of MAML-PINN model")
    
    def _inner_loop(self, theta, support_data_b, support_data_f, alpha, beta):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            theta (List[Tensor]): current model parameters
            support_data_b (Tensor): task support set inputs (boundary data)
                data shape: (number, 1), labels shape: (number, 1)
            support_data_f (Tensor): task support set inputs (pde data)
                data shape: (number, 1), labels shape: (number, 1)
            alpha, beta (float): parameters of the PDE indicating the task

        Returns:
            parameters (List[Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
        """

        inner_loss_b = []
        inner_loss_f = []
        inner_loss   = []

        model_phi = deepcopy(self.model)
        model_phi.load_state_dict(theta)

        loss_fn = torch.nn.MSELoss()
        opt_fn = torch.optim.SGD(model_phi.parameters(), lr=self._inner_lr)

        for _ in range(self._num_inner_steps): 
            opt_fn.zero_grad()
            input_b, target_b = support_data_b
            input_f, target_f = support_data_f
            input_b = input_b.to(self.device)
            target_b = target_b.to(self.device)
            input_f = input_f.to(self.device)
            target_f = target_f.to(self.device)

            loss_b = loss_fn(model_phi(input_b), target_b)
            loss_f = loss_fn(model_phi.calc_loss_f(input_f, target_f, alpha, beta))

            loss_b.to(self.device)
            loss_f.to(self.device)

            loss = loss_b * 10 + loss_f

            loss.backward()
            
            opt_fn.step()

            inner_loss_b += [loss_b.item()]
            inner_loss_f += [loss_f.item()]
            inner_loss   += [loss.item()]

        loss_b = loss_fn(model_phi(input_b), target_b)
        loss_f = loss_fn(model_phi.calc_loss_f(input_f, target_f, alpha, beta))

        inner_loss_b += [loss_b.item()]
        inner_loss_f += [loss_f.item()]
        inner_loss   += [loss_b.item() + loss_f.item()]

        phi = model_phi.state_dict()

        assert phi != None
        assert len(inner_loss) == self._num_inner_steps + 1

        return phi, inner_loss_b, inner_loss_f, inner_loss





