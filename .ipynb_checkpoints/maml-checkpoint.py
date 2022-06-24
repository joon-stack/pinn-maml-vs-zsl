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


LOG_INTERVAL = 20
VAL_INTERVAL = 100
SAVE_INTERVAL = 100

class MAML:
    def __init__(self, num_inner_steps, inner_lr, outer_lr, num_data_i, num_data_b, num_data_f, low, high, eqname='burgers', zero_shot=False, load=False, modelpath=None, savename=None):

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
            self.model = PINN(20, 8, zero_shot=zero_shot, dim=2, param_num=1)
        elif eqname == 'poisson':
            self.model = PINN(20, 5, zero_shot=zero_shot, dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Current device: ", self.device)
        self._save_dir = 'models/maml_{}.data'.format(eqname)
        os.makedirs(self._save_dir, exist_ok=True)
        if load:
            self.model.load_state_dict(torch.load(modelpath))
            print("Model loaded successfully from {}".format(modelpath))
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

        self.savename = savename

        print('Zero shot mode is' , self.zero_shot)

        print("Finished initialization of MAML-PINN model")

        print("Trained model will be saved by {}".format(self.savename))
    
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

        nrmse_batch = []

        model_phi = deepcopy(self.model)
        model_phi.load_state_dict(theta)
        model_phi.to(self.device)
        # print(list(model_phi.parameters())[0][0])

        if train:
            model_phi.train()
        else:
            model_phi.eval()

        loss_fn = torch.nn.MSELoss()
        opt_fn = torch.optim.SGD(model_phi.parameters(), lr=self._inner_lr)


        # generate train data based on (alpha, beta) from support data
        if self.eqname == 'poisson':
            alpha, beta = support
            support_data_b = make_training_boundary_data(self._num_data_b, alpha=alpha, beta=beta, low=self.low, high=self.high, zero_shot=self.zero_shot)
            support_data_f = make_training_domain_data(self._num_data_f, alpha=alpha, beta=beta, low=self.low, high=self.high, zero_shot=self.zero_shot)
        elif self.eqname == 'burgers':
            alpha = support
            num_data_i = self._num_data_i if train else 100
            num_data_b = self._num_data_b if train else 100
            num_data_f = self._num_data_f if train else 1000
            support_data_i = make_training_initial_data_burgers(num_data_i, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)
            support_data_b = make_training_boundary_data_burgers(num_data_b, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)
            support_data_f = make_training_domain_data_burgers(num_data_f, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)

        num_inner_steps = self._num_inner_steps

        if self.eqname == 'poisson':
            for _ in range(num_inner_steps): 
                nrmse = model_phi.validate(alpha, beta, low=self.low, high=self.high) if not train else None
                opt_fn.zero_grad()
                input_b, target_b = support_data_b
                input_f, target_f = support_data_f
                input_b = input_b.to(self.device)
                target_b = target_b.to(self.device)
                input_f = input_f.to(self.device)
                target_f = target_f.to(self.device)

                loss_b = loss_fn(model_phi(input_b), target_b)
                loss_f = model_phi.calc_loss_f(input_f, target_f, alpha, beta)

                loss = loss_b * 10 + loss_f

                loss.backward()
                    
                opt_fn.step()

                inner_loss_b += [loss_b.item()]
                inner_loss_f += [loss_f.item()]
                inner_loss   += [loss.item()]
                
                if not train:
                    nrmse_batch += [nrmse]

            input_b, target_b = support_data_b
            input_f, target_f = support_data_f
            input_b = input_b.to(self.device)
            target_b = target_b.to(self.device)
            input_f = input_f.to(self.device)
            target_f = target_f.to(self.device)
            loss_b = loss_fn(model_phi(input_b), target_b)
            loss_f = model_phi.calc_loss_f(input_f, target_f, alpha, beta)
            loss = loss_b * 10 + loss_f
            inner_loss_b += [loss_b.item()]
            inner_loss_f += [loss_f.item()]
            inner_loss   += [loss.item()]
            grad = torch.autograd.grad(loss, model_phi.parameters()) if train else None
            nrmse = model_phi.validate(alpha, beta, low=self.low, high=self.high) if not train else None
            if not train:
                nrmse_batch += [nrmse]
            phi = model_phi.state_dict()
            # print(inner_loss_b)
 

        elif self.eqname == 'burgers':
            if not train:
                input_f, target_f = support_data_f
                truth = model_phi.get_burgers(alpha=input_f[0].detach().numpy()[-1])
            
            for _ in range(num_inner_steps): 
                nrmse = model_phi.validate_burgers(model_phi, alpha, truth) if not train else None
                if not train:
                    nrmse_batch += [nrmse]


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
            loss = loss_i + loss_b + loss_f
            inner_loss_i += [loss_i.item()]
            inner_loss_b += [loss_b.item()]
            inner_loss_f += [loss_f.item()]
            inner_loss   += [loss.item()]
            grad = torch.autograd.grad(loss, model_phi.parameters()) if train else None
            

            phi = model_phi.state_dict()

            nrmse = model_phi.validate_burgers(model_phi, alpha, truth) if not train else None
            if not train:
                nrmse_batch += [nrmse]

        assert phi != None
        assert len(inner_loss) == num_inner_steps + 1



        return phi, grad, inner_loss_i, inner_loss_b, inner_loss_f, inner_loss, nrmse_batch


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

        grad_sum = [torch.zeros(w.shape).to(self.device) for w in list(self.model.parameters())]

        nrmse_batch = []

        model_outer = deepcopy(self.model)
        model_outer.load_state_dict(theta)
        model_outer.to(self.device)

        loss_fn = nn.MSELoss()

        if self.eqname == 'poisson':
            for task in task_batch:
                support, query = task
                alpha, beta = query
                phi, grad, loss_sup_i, loss_sup_b, loss_sup_f, loss_sup, nrmse = self._inner_loop(theta, support, train)
                inner_loss_b.append(loss_sup_b)
                inner_loss_f.append(loss_sup_f)
                inner_loss.append(loss_sup)

                model_outer.load_state_dict(phi)
                
                query_data_b = make_training_boundary_data(self._num_data_b, alpha=alpha, beta=beta, low=self.low, high=self.high, zero_shot=self.zero_shot)
                query_data_f = make_training_domain_data(self._num_data_f, alpha=alpha, beta=beta, low=self.low, high=self.high, zero_shot=self.zero_shot)
                
                input_b, target_b = query_data_b
                input_f, target_f = query_data_f
                input_b = input_b.to(self.device)
                target_b = target_b.to(self.device)
                input_f = input_f.to(self.device)
                target_f = target_f.to(self.device)

                loss_b = loss_fn(model_outer(input_b), target_b)
                loss_f = model_outer.calc_loss_f(input_f, target_f, alpha, beta)

                loss = loss_b * 10 + loss_f

                grad = torch.autograd.grad(loss, model_outer.parameters()) if train else None

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
            
            return np.mean(inner_loss, axis=0), np.mean(inner_loss_b, axis=0), np.mean(inner_loss_f, axis=0), np.mean(nrmse_batch, axis=0)
        
        elif self.eqname == 'burgers':
            for task in task_batch:
                support, query = task
                alpha = query
                phi, grad, loss_sup_i, loss_sup_b, loss_sup_f, loss_sup, nrmse = self._inner_loop(theta, support, train)
                inner_loss_i.append(loss_sup_i)
                inner_loss_b.append(loss_sup_b)
                inner_loss_f.append(loss_sup_f)
                inner_loss.append(loss_sup)

                model_outer.load_state_dict(phi)
                
                query_data_i = make_training_initial_data_burgers(self._num_data_i, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)
                query_data_b = make_training_boundary_data_burgers(self._num_data_b, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)
                query_data_f = make_training_domain_data_burgers(self._num_data_f, alpha=alpha, low=self.low, high=self.high, zero_shot=self.zero_shot)
                
                input_i, target_i = query_data_i
                input_b, target_b = query_data_b
                input_f, target_f = query_data_f
                input_i = input_i.to(self.device)
                target_i = target_i.to(self.device)
                input_b = input_b.to(self.device)
                target_b = target_b.to(self.device)
                input_f = input_f.to(self.device)
                target_f = target_f.to(self.device)

                loss_i = loss_fn(model_outer(input_i), target_i)
                loss_b = loss_fn(model_outer(input_b), target_b)
                loss_f = model_outer.calc_loss_f_burgers(input_f, target_f, alpha)

                loss = loss_i + loss_b + loss_f

                grad = torch.autograd.grad(loss, model_outer.parameters()) if train else None

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

            return np.mean(inner_loss, axis=0), np.mean(inner_loss_i, axis=0), np.mean(inner_loss_b, axis=0), np.mean(inner_loss_f, axis=0), np.mean(nrmse_batch, axis=0)
    
    
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
                            'inner_loss_pre_adapt': [],
                            'inner_loss_b_pre_adapt': [],
                            'inner_loss_f_pre_adapt': [],
                            'inner_loss': [],
                            'inner_loss_b': [],
                            'inner_loss_f': [],
                        }

            val_ood_loss = {
                            'inner_loss_pre_adapt': [],
                            'inner_loss_b_pre_adapt': [],
                            'inner_loss_f_pre_adapt': [],
                            'inner_loss': [],
                            'inner_loss_b': [],
                            'inner_loss_f': []
                        }
            
            nrmse = {
                        'nrmse_val': [],
                        'nrmse_val_ood': [],
                        'nrmse_val_pre_adapt': [],
                        'nrmse_val_ood_pre_adapt': [],
                    }

            val_task = generate_task(num_val_tasks)
            val_ood_task = generate_task(num_val_tasks, ood=True)

            # print(val_task)
            # print(val_ood_task)

            inner_loss_val, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
            print("Validation before training Pre-Adapt({3:.3f}, {4:.3f})| Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, inner_loss_val_b[0], inner_loss_val_f[0], val_task[0][0][0], val_task[0][0][1], inner_loss_val[0], nrmse_val[0]))

            val_loss['inner_loss_pre_adapt'].append(inner_loss_val[0])
            val_loss['inner_loss_b_pre_adapt'].append(inner_loss_val_b[0])
            val_loss['inner_loss_f_pre_adapt'].append(inner_loss_val_f[0])

            nrmse['nrmse_val_pre_adapt'].append(nrmse_val[0])

            inner_loss_val, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
            print("Validation before training Post-Adapt({3:.3f}, {4:.3f})| Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, inner_loss_val_b[-1], inner_loss_val_f[-1], val_task[0][0][0], val_task[0][0][1], inner_loss_val[-1], nrmse_val[-1]))

            val_loss['inner_loss'].append(inner_loss_val[-1])
            val_loss['inner_loss_b'].append(inner_loss_val_b[-1])
            val_loss['inner_loss_f'].append(inner_loss_val_f[-1])

            nrmse['nrmse_val'].append(nrmse_val[-1])
                
            # out-of-distribution
            inner_loss_val_ood, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
            print("Validation OOD before training Pre-Adapt({3:.3f}, {4:.3f}) | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, inner_loss_val_ood_b[0], inner_loss_val_ood_f[0], val_ood_task[0][0][0], val_ood_task[0][0][1], inner_loss_val_ood[0], nrmse_val_ood[0]))

            val_ood_loss['inner_loss_pre_adapt'].append(inner_loss_val_ood[0])
            val_ood_loss['inner_loss_b_pre_adapt'].append(inner_loss_val_ood_b[0])
            val_ood_loss['inner_loss_f_pre_adapt'].append(inner_loss_val_ood_f[0])

            nrmse['nrmse_val_ood_pre_adapt'].append(nrmse_val_ood[0])

            inner_loss_val_ood, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
            print("Validation OOD before training Post-Adapt({3:.3f}, {4:.3f}) | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, inner_loss_val_ood_b[-1], inner_loss_val_ood_f[-1], val_ood_task[0][0][0], val_ood_task[0][0][1], inner_loss_val_ood[-1], nrmse_val_ood[-1]))

            val_ood_loss['inner_loss'].append(inner_loss_val_ood[-1])
            val_ood_loss['inner_loss_b'].append(inner_loss_val_ood_b[-1])
            val_ood_loss['inner_loss_f'].append(inner_loss_val_ood_f[-1])

            nrmse['nrmse_val_ood'].append(nrmse_val_ood[-1])

            for i in range(1, train_steps + 1):
                self._train_step += 1
                train_task = generate_task(num_train_tasks)
                inner_loss, inner_loss_b, inner_loss_f, _ = self._outer_loop(train_task, train=True)

                train_loss['inner_loss'].append(inner_loss)
                train_loss['inner_loss_b'].append(inner_loss_b)
                train_loss['inner_loss_f'].append(inner_loss_f)

                # print(list(self.model.parameters())[0][0][1]])
                
                if i % SAVE_INTERVAL == 0:
                    print("Model saved")
                    torch.save(self.model.state_dict(), 'maml_poisson_high{}_{}.data'.format(self.high, i))

                if i % LOG_INTERVAL == 0:
                    
                    print("Step {0} Pre-Adapt | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {3:.4f}"
                    .format(self._train_step, inner_loss_b[0], inner_loss_f[0], inner_loss[0]))

                    print("Step {0} Post-Adapt | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {3:.4f}"
                    .format(self._train_step, inner_loss_b[-1], inner_loss_f[-1], inner_loss[-1]))

                if i % VAL_INTERVAL == 0:
                    
                    # in-distibution
                    # val_task = generate_task(1)
                    # val_task = [(0.5, 0.5)]
                    inner_loss_val, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
                    print("Validation before training Pre-Adapt({3:.3f}, {4:.3f})| Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
                    .format(self._train_step, inner_loss_val_b[0], inner_loss_val_f[0], val_task[0][0][0], val_task[0][0][1], inner_loss_val[0], nrmse_val[0]))

                    
                    val_loss['inner_loss_pre_adapt'].append(inner_loss_val[0])
                    val_loss['inner_loss_b_pre_adapt'].append(inner_loss_val_b[0])
                    val_loss['inner_loss_f_pre_adapt'].append(inner_loss_val_f[0])

                    nrmse['nrmse_val_pre_adapt'].append(nrmse_val[0])

                    print("Validation before training Post-Adapt({3:.3f}, {4:.3f})| Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
                    .format(self._train_step, inner_loss_val_b[-1], inner_loss_val_f[-1], val_task[0][0][0], val_task[0][0][1], inner_loss_val[-1], nrmse_val[-1]))

                    val_loss['inner_loss'].append(inner_loss_val[-1])
                    val_loss['inner_loss_b'].append(inner_loss_val_b[-1])
                    val_loss['inner_loss_f'].append(inner_loss_val_f[-1])

                    nrmse['nrmse_val'].append(nrmse_val[-1])
                        
                    # out-of-distribution
                    inner_loss_val_ood, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
                    print("Validation OOD before training Pre-Adapt({3:.3f}, {4:.3f}) | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
                    .format(self._train_step, inner_loss_val_ood_b[0], inner_loss_val_ood_f[0], val_ood_task[0][0][0], val_ood_task[0][0][1], inner_loss_val_ood[0], nrmse_val_ood[0]))

                    val_ood_loss['inner_loss_pre_adapt'].append(inner_loss_val_ood[0])
                    val_ood_loss['inner_loss_b_pre_adapt'].append(inner_loss_val_ood_b[0])
                    val_ood_loss['inner_loss_f_pre_adapt'].append(inner_loss_val_ood_f[0])

                    nrmse['nrmse_val_ood_pre_adapt'].append(nrmse_val_ood[0])

                    inner_loss_val_ood, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
                    print("Validation OOD before training Post-Adapt({3:.3f}, {4:.3f}) | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
                    .format(self._train_step, inner_loss_val_ood_b[-1], inner_loss_val_ood_f[-1], val_ood_task[0][0][0], val_ood_task[0][0][1], inner_loss_val_ood[-1], nrmse_val_ood[-1]))

                    val_ood_loss['inner_loss'].append(inner_loss_val_ood[-1])
                    val_ood_loss['inner_loss_b'].append(inner_loss_val_ood_b[-1])
                    val_ood_loss['inner_loss_f'].append(inner_loss_val_ood_f[-1])

                    nrmse['nrmse_val_ood'].append(nrmse_val_ood[-1])
                    
            return train_loss, val_loss, val_ood_loss, nrmse, self.model
        
        elif self.eqname == 'burgers':
            train_loss = {

                            'inner_loss': [],
                            'inner_loss_i': [],
                            'inner_loss_b': [],
                            'inner_loss_f': [],
                            'inner_loss_pre_adapt': [],
                            'inner_loss_i_pre_adapt': [],
                            'inner_loss_b_pre_adapt': [],
                            'inner_loss_f_pre_adapt': []
                        }

            val_loss = {

                            'inner_loss': [],
                            'inner_loss_i': [],
                            'inner_loss_b': [],
                            'inner_loss_f': [],
                            'inner_loss_pre_adapt': [],
                            'inner_loss_i_pre_adapt': [],
                            'inner_loss_b_pre_adapt': [],
                            'inner_loss_f_pre_adapt': []
                        }

            val_ood_loss = {
                            'inner_loss': [],
                            'inner_loss_i': [],
                            'inner_loss_b': [],
                            'inner_loss_f': [],
                            'inner_loss_pre_adapt': [],
                            'inner_loss_i_pre_adapt': [],
                            'inner_loss_b_pre_adapt': [],
                            'inner_loss_f_pre_adapt': []
                        }
            
            nrmse = {
                        'nrmse_val': [],
                        'nrmse_val_ood': [],
                        'nrmse_val_pre_adapt': [],
                        'nrmse_val_ood_pre_adapt': []
                    }

            # val_task = [(0.01 / np.pi, 0.02 / np.pi)]
            
            val_task = generate_task_burgers(num_val_tasks)
            # val_ood_task = [(0.2 / np.pi, 0.3 / np.pi)]
            val_ood_task = generate_task_burgers(num_val_tasks, ood=True)

            train_task = generate_task_burgers(num_train_tasks)

            print(val_task)
            print(val_ood_task)

            inner_loss_val, inner_loss_val_i, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
            print("Validation before training Pre-Adaptation ({3:.3f})| Inner_loss_I: {4:.4f} | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, inner_loss_val_b[0], inner_loss_val_f[0], val_task[0][1], inner_loss_val_i[0], inner_loss_val[0], nrmse_val[0]))

            val_loss['inner_loss_pre_adapt'].append(inner_loss_val[0])
            val_loss['inner_loss_i_pre_adapt'].append(inner_loss_val_i[0])
            val_loss['inner_loss_b_pre_adapt'].append(inner_loss_val_b[0])
            val_loss['inner_loss_f_pre_adapt'].append(inner_loss_val_f[0])

            nrmse['nrmse_val_pre_adapt'].append(nrmse_val[0])


            print("Validation before training ({3:.3f})| Inner_loss_I: {4:.4f} | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {5:.4f} | NRMSE: {6:.4f}"
            .format(self._train_step, inner_loss_val_b[-1], inner_loss_val_f[-1], val_task[0][1], inner_loss_val_i[-1], inner_loss_val[-1], nrmse_val[-1]))

            val_loss['inner_loss'].append(inner_loss_val[-1])
            val_loss['inner_loss_i'].append(inner_loss_val_i[-1])
            val_loss['inner_loss_b'].append(inner_loss_val_b[-1])
            val_loss['inner_loss_f'].append(inner_loss_val_f[-1])

            nrmse['nrmse_val'].append(nrmse_val[-1])

            inner_loss_val_ood, inner_loss_val_ood_i, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
            print("validation OOD before training Pre-Adaptation ({3:.3f})| Inner_loss_I: {6:.4f} |Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f} | NRMSE: {5:.4f}"
            .format(self._train_step, inner_loss_val_ood_b[0], inner_loss_val_ood_f[0], val_ood_task[0][1], inner_loss_val_ood[0], nrmse_val_ood[0], inner_loss_val_ood_i[0]))
            val_ood_loss['inner_loss_pre_adapt'].append(inner_loss_val_ood[0])
            val_ood_loss['inner_loss_i_pre_adapt'].append(inner_loss_val_ood_i[0])
            val_ood_loss['inner_loss_b_pre_adapt'].append(inner_loss_val_ood_b[0])
            val_ood_loss['inner_loss_f_pre_adapt'].append(inner_loss_val_ood_f[0])

            nrmse['nrmse_val_ood_pre_adapt'].append(nrmse_val_ood[0])

            print("validation OOD before training ({3:.3f})| Inner_loss_I: {6:.4f} |Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f} | NRMSE: {5:.4f}"
            .format(self._train_step, inner_loss_val_ood_b[-1], inner_loss_val_ood_f[-1], val_ood_task[0][1], inner_loss_val_ood[-1], nrmse_val_ood[-1], inner_loss_val_ood_i[-1]))
            val_ood_loss['inner_loss'].append(inner_loss_val_ood[-1])
            val_ood_loss['inner_loss_i'].append(inner_loss_val_ood_i[-1])
            val_ood_loss['inner_loss_b'].append(inner_loss_val_ood_b[-1])
            val_ood_loss['inner_loss_f'].append(inner_loss_val_ood_f[-1])

            nrmse['nrmse_val_ood'].append(nrmse_val_ood[-1])

            for i in range(1, train_steps + 1):
                self._train_step += 1
                # train_task = generate_task_burgers(num_train_tasks)
                inner_loss, inner_loss_i, inner_loss_b, inner_loss_f, _ = self._outer_loop(train_task, train=True)

                train_loss['inner_loss'].append(inner_loss)
                train_loss['inner_loss_i'].append(inner_loss_i)
                train_loss['inner_loss_b'].append(inner_loss_b)
                train_loss['inner_loss_f'].append(inner_loss_f)

                train_loss['inner_loss_pre_adapt'].append(inner_loss)
                train_loss['inner_loss_i_pre_adapt'].append(inner_loss_i)
                train_loss['inner_loss_b_pre_adapt'].append(inner_loss_b)
                train_loss['inner_loss_f_pre_adapt'].append(inner_loss_f)

                # print(list(self.model.parameters())[0][0])
                
                if i % SAVE_INTERVAL == 0:
                    print("Model saved")
                    fname = 'models/{}_{}.data'.format(self.savename, i)
                    torch.save(self.model.state_dict(), fname)

                if i % LOG_INTERVAL == 0:
                    
                    print("Step {0} Pre-Adapt | Inner_loss_I: {3:.4f} | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f}"
                    .format(self._train_step, inner_loss_b[0], inner_loss_f[0], inner_loss_i[0], inner_loss[0]))
                    print("Step {0} Post-Adapt | Inner_loss_I: {3:.4f} | Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f}"
                    .format(self._train_step, inner_loss_b[-1], inner_loss_f[-1], inner_loss_i[-1], inner_loss[-1]))
                
                if i % VAL_INTERVAL == 0:
                    
                    # in-distibution
                    # val_task = generate_task_burgers(1)
                    # val_task = [(0.5, 0.5)]
                    inner_loss_val, inner_loss_val_i, inner_loss_val_b, inner_loss_val_f, nrmse_val = self._outer_loop(val_task, train=False)
                    print("Validation Pre-adaptation ({3:.3f})| Inner_loss_I: {6:.4f} |Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f} | NRMSE: {5:.4f}"
                    .format(self._train_step, inner_loss_val_b[0], inner_loss_val_f[0], val_task[0][1], inner_loss_val[0], nrmse_val[0], inner_loss_val_i[0]))
                    val_loss['inner_loss_pre_adapt'].append(inner_loss_val[0])
                    val_loss['inner_loss_i_pre_adapt'].append(inner_loss_val_i[0])
                    val_loss['inner_loss_b_pre_adapt'].append(inner_loss_val_b[0])
                    val_loss['inner_loss_f_pre_adapt'].append(inner_loss_val_f[0])
                    nrmse['nrmse_val_pre_adapt'].append(nrmse_val[0])

                    print("Validation Post-adaptation ({3:.3f})| Inner_loss_I: {6:.4f} |Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f} | NRMSE: {5:.4f}"
                    .format(self._train_step, inner_loss_val_b[-1], inner_loss_val_f[-1], val_task[0][1], inner_loss_val[-1], nrmse_val[-1], inner_loss_val_i[-1]))
                    val_loss['inner_loss'].append(inner_loss_val[-1])
                    val_loss['inner_loss_i'].append(inner_loss_val_i[-1])
                    val_loss['inner_loss_b'].append(inner_loss_val_b[-1])
                    val_loss['inner_loss_f'].append(inner_loss_val_f[-1])
                    nrmse['nrmse_val'].append(nrmse_val[-1])
                        
                    # out-of-distribution
                    # val_ood_task = generate_task(1, ood=True)
                    # val_ood_task = [(1.3, 1.3)]
                    inner_loss_val_ood, inner_loss_val_ood_i, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
                    print("validation OOD Pre-Adaptation ({3:.3f})| Inner_loss_I: {6:.4f} |Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f} | NRMSE: {5:.4f}"
                    .format(self._train_step, inner_loss_val_ood_b[0], inner_loss_val_ood_f[0], val_ood_task[0][1], inner_loss_val_ood[0], nrmse_val_ood[0], inner_loss_val_ood_i[0]))
                    val_ood_loss['inner_loss_pre_adapt'].append(inner_loss_val_ood[0])
                    val_ood_loss['inner_loss_i_pre_adapt'].append(inner_loss_val_ood_i[0])
                    val_ood_loss['inner_loss_b_pre_adapt'].append(inner_loss_val_ood_b[0])
                    val_ood_loss['inner_loss_f_pre_adapt'].append(inner_loss_val_ood_f[0])
                    nrmse['nrmse_val_ood_pre_adapt'].append(nrmse_val[0])

                    inner_loss_val_ood, inner_loss_val_ood_i, inner_loss_val_ood_b, inner_loss_val_ood_f, nrmse_val_ood = self._outer_loop(val_ood_task, train=False)
                    print("validation OOD Post-Adaptation ({3:.3f})| Inner_loss_I: {6:.4f} |Inner_loss_B: {1:.4f} | Inner_loss_F: {2:.4f} | Inner_loss: {4:.4f} | NRMSE: {5:.4f}"
                    .format(self._train_step, inner_loss_val_ood_b[-1], inner_loss_val_ood_f[-1], val_ood_task[0][1], inner_loss_val_ood[-1], nrmse_val_ood[-1], inner_loss_val_ood_i[-1]))
                    val_ood_loss['inner_loss'].append(inner_loss_val_ood[-1])
                    val_ood_loss['inner_loss_i'].append(inner_loss_val_ood_i[-1])
                    val_ood_loss['inner_loss_b'].append(inner_loss_val_ood_b[-1])
                    val_ood_loss['inner_loss_f'].append(inner_loss_val_ood_f[-1])
                    nrmse['nrmse_val_ood'].append(nrmse_val[-1])
            
            return train_loss, val_loss, val_ood_loss, nrmse, self.model


def main():
    maml = MAML(5, 0.01, 0.0001, 0, 2, 1, low=-1, high=1, eqname='poisson')
    train_loss, val_loss, val_ood_loss, nrmse, model = maml.train(1000, 1000, 5)
    print(train_loss['inner_loss'])

if __name__ == "__main__":
    main()

            

            

        




