import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os


from model import *
from generate_data import *
from utils import *
from copy import deepcopy

from burgers import *


LOG_INTERVAL = 1
VAL_INTERVAL = 50
SAVE_INTERVAL = 100

class MAML:
    def __init__(self, num_inner_steps, inner_lr, outer_lr, num_data_i, num_data_b, num_data_f, low, high, eqname='burgers', zero_shot=False):

        """Initializes First-Order Model-Agnostic Meta-Learning to train Physics-Informed Neural Networks.

        Args:
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
            outer_lr (float): learning rate for outer-loop optimization
            num_data_b (int): number of boundary data
            num_data_f (int): number of PDE data
        """

        print("Initializing MAML-PINN model")

        if eqname == 'burgers':
            self.model = PINN(20, 5, zero_shot=zero_shot, dim=2)
        elif eqname == 'poisson':
            self.model = PINN(20, 5, zero_shot=zero_shot, dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Current device: ", self.device)
        self._save_dir = 'models/maml_{}.data'.format(eqname)
        os.makedirs(self._save_dir, exist_ok=True)
        self.model.to(self.device)
        print(self.model.device)

        self._num_inner_steps = num_inner_steps
        
        self._inner_lr = inner_lr
        self._outer_lr = outer_lr
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self._outer_lr)

        self._num_data_i = num_data_i
        self._num_data_b = num_data_b
        self._num_data_f = num_data_f

        self._train_step = 0

        self.low = low
        self.high = high

        self.eqname = eqname

        self.zero_shot = zero_shot

        print("Finished initialization of MAML-PINN model")
    
    def _inner_loop(self, theta, support, train=True):

        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            theta (List[Tensor]): current model parameters
            support (Tensor): support task. (alpha, beta)
            train (Boolean): whether the model is trained or not, 
                             if true, it returns gradient

        Returns:
            parameters (phi) (List[Tensor]): adapted network parameters
            inner_loss_b (list[float]): support set loss_b over the course of
                the inner loop, length num_inner_steps + 1
            inner_loss_f (list[float]): support set loss_f over the course of
                the inner loop, length num_inner_steps + 1
            inner_loss (list[float]): support set loss over the course of
                the inner loop, length num_inner_steps + 1
            grad (list[Tensor]): gradient of loss w.r.t. phi
        """

        inner_loss_i = []
        inner_loss_b = []
        inner_loss_f = []
        inner_loss   = []

        model_phi = deepcopy(self.model)
        model_phi.load_state_dict(theta)
        model_phi.to(self.device)

        loss_fn = torch.nn.MSELoss()
        opt_fn = torch.optim.SGD(model_phi.parameters(), lr=self._inner_lr)


        # generate train data based on (alpha, beta) from support data
        if self.eqname == 'poisson':
            alpha, beta = support
            support_data_b = make_training_boundary_data(self._num_data_b, alpha=alpha, beta=beta, low=self.low, high=self.high, zero_shot=self.zero_shot)
            support_data_f = make_training_domain_data(self._num_data_f, alpha=alpha, beta=beta, low=self.low, high=self.high, zero_shot=self.zero_shot)
        elif self.eqname == 'burgers':
            alpha = support
            support_data_i = make_training_initial_data_burgers(self._num_data_i, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)
            support_data_b = make_training_boundary_data_burgers(self._num_data_b, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)
            support_data_f = make_training_domain_data_burgers(self._num_data_f, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)

        if self.eqname == 'poisson':
            for _ in range(self._num_inner_steps): 
                opt_fn.zero_grad()
                input_b, target_b = support_data_b
                input_f, target_f = support_data_f
                input_b = input_b.to(self.device)
                target_b = target_b.to(self.device)
                input_f = input_f.to(self.device)
                target_f = target_f.to(self.device)

                loss_b = loss_fn(model_phi(input_b), target_b)
                loss_f = model_phi.calc_loss_f(input_f, target_f, alpha, beta)

                loss_b.to(self.device)
                loss_f.to(self.device)

                loss = loss_b * 10 + loss_f

                loss.backward()
                
                opt_fn.step()

                inner_loss_b += [loss_b.item()]
                inner_loss_f += [loss_f.item()]
                inner_loss   += [loss.item()]

            loss_b = loss_fn(model_phi(input_b), target_b)
            loss_f = model_phi.calc_loss_f(input_f, target_f, alpha, beta)
            loss = loss_b * 10 + loss_f
            inner_loss_b += [loss_b.item()]
            inner_loss_f += [loss_f.item()]
            inner_loss   += [loss.item()]
            grad = torch.autograd.grad(loss, model_phi.parameters()) if train else None

            phi = model_phi.state_dict()

            
            nrmse = model_phi.validate(alpha, beta) if not train else None

        elif self.eqname == 'burgers':
            for _ in range(self._num_inner_steps): 
                opt_fn.zero_grad()
                input_i, target_i = support_data_i
                input_b, target_b = support_data_b
                input_f, target_f = support_data_f
                input_i = input_i.to(self.device)
                target_i = target_i.to(self.device)
                input_b = input_b.to(self.device)
                target_b = target_b.to(self.device)
                input_f = input_f.to(self.device)
                target_f = target_f.to(self.device)

                loss_i = loss_fn(model_phi(input_i), target_i)
                loss_b = loss_fn(model_phi(input_b), target_b)
                loss_f = model_phi.calc_loss_f_burgers(input_f, target_f, alpha)

                loss_i.to(self.device)
                loss_b.to(self.device)
                loss_f.to(self.device)

                loss = loss_i + loss_b + loss_f

                loss.backward()
                
                opt_fn.step()

                inner_loss_i += [loss_i.item()]
                inner_loss_b += [loss_b.item()]
                inner_loss_f += [loss_f.item()]
                inner_loss   += [loss.item()]

            loss_i = loss_fn(model_phi(input_i), target_i)
            loss_b = loss_fn(model_phi(input_b), target_b)
            loss_f = model_phi.calc_loss_f_burgers(input_f, target_f, alpha)
            loss = loss_i + loss_b + loss_f
            inner_loss_i += [loss_i.item()]
            inner_loss_b += [loss_b.item()]
            inner_loss_f += [loss_f.item()]
            inner_loss   += [loss.item()]
            grad = torch.autograd.grad(loss, model_phi.parameters()) if train else None

            phi = model_phi.state_dict()

            # nrmse = None
            # nrmse = model_phi.validate(alpha, beta) if not train else None
            nrmse = model_phi.validate_burgers(alpha) if not train else None

        assert phi != None
        assert len(inner_loss) == self._num_inner_steps + 1


        return phi, grad, inner_loss_i, inner_loss_b, inner_loss_f, inner_loss, nrmse


    def _outer_loop(self, task_batch, train=None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from PDE 
            each task consists with (support, query)
            each support and query consists with (alpha, beta)
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batchk
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """

        theta = self.model.state_dict()

        inner_loss_i = []
        inner_loss_b = []
        inner_loss_f = []
        inner_loss   = []

        model_outer = deepcopy(self.model)
        model_outer.load_state_dict(theta)
        model_outer.to(self.device)

        grad_sum = [torch.zeros(w.shape).to(self.device) for w in list(model_outer.parameters())]

        nrmse_batch = []

        if self.eqname == 'poisson':
            for task in task_batch:
                support = task
                phi, grad, loss_sup_i, loss_sup_b, loss_sup_f, loss_sup, nrmse = self._inner_loop(theta, support, train)
                inner_loss_b.append(loss_sup_b[-1])
                inner_loss_f.append(loss_sup_f[-1])
                inner_loss.append(loss_sup[-1])
                model_outer.load_state_dict(phi)

                if train:
                    for g_sum, g in zip(grad_sum, grad):
                        g_sum += g
                else:
                    nrmse_batch += [nrmse]

            if train:
                for g in grad_sum:
                    g /= len(task_batch)
                for w, g in zip(list(self.model.parameters()), grad_sum):
                    w.grad = g
                self._optimizer.step()

            return np.mean(inner_loss), np.mean(inner_loss_b), np.mean(inner_loss_f), nrmse_batch
        
        elif self.eqname == 'burgers':
            for task in task_batch:
                support = task
                phi, grad, loss_sup_i, loss_sup_b, loss_sup_f, loss_sup, nrmse = self._inner_loop(theta, support, train)
                inner_loss_i.append(loss_sup_i[-1])
                inner_loss_b.append(loss_sup_b[-1])
                inner_loss_f.append(loss_sup_f[-1])
                inner_loss.append(loss_sup[-1])
                model_outer.load_state_dict(phi)

                if train:
                    for g_sum, g in zip(grad_sum, grad):
                        g_sum += g
                else:
                    nrmse_batch += [nrmse]
            if train:
                for g in grad_sum:
                    g /= len(task_batch)
                for w, g in zip(list(self.model.parameters()), grad_sum):
                    w.grad = g
                self._optimizer.step()

            return np.mean(inner_loss), np.mean(inner_loss_i), np.mean(inner_loss_b), np.mean(inner_loss_f), nrmse_batch
    
    
    def train(self, train_steps, num_train_tasks, num_val_tasks):
        """Train the MAML.

        Optimizes MAML meta-parameters

        Args:
            train_steps (int): the number of steps this model should train for
        """
        print("Start MAML training at iteration {}".format(self._train_step))

        if self.eqname == 'poisson':

            train_loss = {

                            'inner_loss': [],
                            'inner_loss_b': [],
                            'inner_loss_f': []
                        }

            val_loss = {

                            'inner_loss': [],
                            'inner_loss_b': [],
                            'inner_loss_f': []
                        }

            val_ood_loss = {
                            'inner_loss': [],
                            'inner_loss_b': [],
                            'inner_loss_f': []
                        }
            
            nrmse = {
                        'nrmse_val': [],
                        'nrmse_val_ood': []
                    }

            val_task = generate_task(num_val_tasks)
            val_ood_task = generate_task(num_val_tasks, ood=True)

            print(val_task)
            print(val_ood_task)

            inner_loss_val, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
            print("Validation before training ({3:.3f}, {4:.3f})| Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, np.mean(inner_loss_val_b), np.mean(inner_loss_val_f), val_task[0][0], val_task[0][1], np.mean(inner_loss_val), np.mean(nrmse_val)))

            val_loss['inner_loss'].append(inner_loss_val)
            val_loss['inner_loss_b'].append(inner_loss_val_b)
            val_loss['inner_loss_f'].append(inner_loss_val_f)

            nrmse['nrmse_val'].append(nrmse_val)
                
            # out-of-distribution
            inner_loss_val_ood, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
            print("Validation OOD before training ({3:.3f}, {4:.3f}) | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, np.mean(inner_loss_val_ood_b), np.mean(inner_loss_val_ood_f), val_ood_task[0][0], val_ood_task[0][1], np.mean(inner_loss_val_ood), np.mean(nrmse_val_ood)))

            val_ood_loss['inner_loss'].append(inner_loss_val_ood)
            val_ood_loss['inner_loss_b'].append(inner_loss_val_ood_b)
            val_ood_loss['inner_loss_f'].append(inner_loss_val_ood_f)

            nrmse['nrmse_val_ood'].append(nrmse_val_ood)

            for i in range(1, train_steps + 1):
                self._train_step += 1
                train_task = generate_task(num_train_tasks)
                inner_loss, inner_loss_b, inner_loss_f, _ = self._outer_loop(train_task, train=True)

                train_loss['inner_loss'].append(inner_loss)
                train_loss['inner_loss_b'].append(inner_loss_b)
                train_loss['inner_loss_f'].append(inner_loss_f)

                # print(list(self.model.parameters())[0][0])
                
                if i % SAVE_INTERVAL == 0:
                    print("Model saved")
                    torch.save(self.model.state_dict(), 'maml.data')

                if i % LOG_INTERVAL == 0:
                    
                    print("Step {0} | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f}"
                    .format(self._train_step, np.mean(inner_loss_b), np.mean(inner_loss_f)))
                
                if i % VAL_INTERVAL == 0:
                    
                    # in-distibution
                    # val_task = generate_task(1)
                    # val_task = [(0.5, 0.5)]
                    inner_loss_val, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
                    print("Validation ({3:.3f}, {4:.3f})| Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
                    .format(self._train_step, np.mean(inner_loss_val_b), np.mean(inner_loss_val_f), val_task[0][0], val_task[0][1], np.mean(inner_loss_val), np.mean(nrmse_val)))
                    val_loss['inner_loss'].append(inner_loss_val)
                    val_loss['inner_loss_b'].append(inner_loss_val_b)
                    val_loss['inner_loss_f'].append(inner_loss_val_f)

                    nrmse['nrmse_val'].append(nrmse_val)
                        
                    # out-of-distribution
                    # val_ood_task = generate_task(1, ood=True)
                    # val_ood_task = [(1.3, 1.3)]
                    inner_loss_val_ood, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
                    print("Validation OOD ({3:.3f}, {4:.3f}) | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
                    .format(self._train_step, np.mean(inner_loss_val_ood_b), np.mean(inner_loss_val_ood_f), val_ood_task[0][0], val_ood_task[0][1], np.mean(inner_loss_val_ood), np.mean(nrmse_val_ood)))

                    val_ood_loss['inner_loss'].append(inner_loss_val_ood)
                    val_ood_loss['inner_loss_b'].append(inner_loss_val_ood_b)
                    val_ood_loss['inner_loss_f'].append(inner_loss_val_ood_f)

                    nrmse['nrmse_val_ood'].append(nrmse_val_ood)
            
            return train_loss, val_loss, val_ood_loss, nrmse, self.model
        
        elif self.eqname == 'burgers':
            train_loss = {

                            'inner_loss': [],
                            'inner_loss_i': [],
                            'inner_loss_b': [],
                            'inner_loss_f': []
                        }

            val_loss = {

                            'inner_loss': [],
                            'inner_loss_i': [],
                            'inner_loss_b': [],
                            'inner_loss_f': []
                        }

            val_ood_loss = {
                            'inner_loss': [],
                            'inner_loss_i': [],
                            'inner_loss_b': [],
                            'inner_loss_f': []
                        }
            
            nrmse = {
                        'nrmse_val': [],
                        'nrmse_val_ood': []
                    }

            val_task = [(0.01 / np.pi)]
            # val_ood_task = generate_task(num_val_tasks, ood=True)

            print(val_task)
            # print(val_ood_task)

            inner_loss_val, inner_loss_val_i, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
            print("Validation before training ({3:.3f})| Inner_loss_I: {4:.4f} | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, np.mean(inner_loss_val_b), np.mean(inner_loss_val_f), val_task[0], np.mean(inner_loss_val_i), np.mean(inner_loss_val), np.mean(nrmse_val)))

            val_loss['inner_loss'].append(inner_loss_val)
            val_loss['inner_loss_i'].append(inner_loss_val_i)
            val_loss['inner_loss_b'].append(inner_loss_val_b)
            val_loss['inner_loss_f'].append(inner_loss_val_f)

            nrmse['nrmse_val'].append(nrmse_val)

            for i in range(1, train_steps + 1):
                self._train_step += 1
                train_task = generate_task_burgers(num_train_tasks)
                inner_loss, inner_loss_i, inner_loss_b, inner_loss_f, _ = self._outer_loop(train_task, train=True)

                train_loss['inner_loss'].append(inner_loss)
                train_loss['inner_loss_i'].append(inner_loss_i)
                train_loss['inner_loss_b'].append(inner_loss_b)
                train_loss['inner_loss_f'].append(inner_loss_f)

                # print(list(self.model.parameters())[0][0])
                
                if i % SAVE_INTERVAL == 0:
                    print("Model saved")
                    torch.save(self.model.state_dict(), 'models/maml_burgers_{}.data'.format(i))

                if i % LOG_INTERVAL == 0:
                    
                    print("Step {0} | Inner_loss_I: {3:.4f} | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f}"
                    .format(self._train_step, np.mean(inner_loss_b), np.mean(inner_loss_f), np.mean(inner_loss_i)))
                
                if i % VAL_INTERVAL == 0:
                    
                    # in-distibution
                    # val_task = generate_task_burgers(1)
                    # val_task = [(0.5, 0.5)]
                    inner_loss_val, inner_loss_val_i, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
                    print("Validation ({3:.3f})| Inner_loss_I: {6:.4f} |Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f} | NRMSE: {5:.4f}"
                    .format(self._train_step, np.mean(inner_loss_val_b), np.mean(inner_loss_val_f), val_task[0], np.mean(inner_loss_val), np.mean(nrmse_val), np.mean(inner_loss_val_i)))
                    val_loss['inner_loss'].append(inner_loss_val)
                    val_loss['inner_loss_i'].append(inner_loss_val_i)
                    val_loss['inner_loss_b'].append(inner_loss_val_b)
                    val_loss['inner_loss_f'].append(inner_loss_val_f)

                    nrmse['nrmse_val'].append(nrmse_val)
                        
                    # # out-of-distribution
                    # # val_ood_task = generate_task(1, ood=True)
                    # # val_ood_task = [(1.3, 1.3)]
                    # inner_loss_val_ood, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
                    # print("Validation OOD ({3:.3f}, {4:.3f}) | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
                    # .format(self._train_step, np.mean(inner_loss_val_ood_b), np.mean(inner_loss_val_ood_f), val_ood_task[0][0], val_ood_task[0][1], np.mean(inner_loss_val_ood), np.mean(nrmse_val_ood)))

                    # val_ood_loss['inner_loss'].append(inner_loss_val_ood)
                    # val_ood_loss['inner_loss_b'].append(inner_loss_val_ood_b)
                    # val_ood_loss['inner_loss_f'].append(inner_loss_val_ood_f)

                    # nrmse['nrmse_val_ood'].append(nrmse_val_ood)
            
            return train_loss, val_loss, val_ood_loss, nrmse, self.model


def main():
    maml = MAML(5, 0.01, 0.0001, 0, 2, 1, low=-1, high=1, eqname='poisson')
    train_loss, val_loss, val_ood_loss, nrmse, model = maml.train(1000, 1000, 5)
    print(train_loss['inner_loss'])

if __name__ == "__main__":
    main()

            

            

        




