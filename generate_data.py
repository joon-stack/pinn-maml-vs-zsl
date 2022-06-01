import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time

from copy import copy

# tag: -1 for ICs, 0 for BCs, 1 for domain
# solve u_t + u*u_x - (0.01/pi)u_xx = 0, -1 <= x <= 1, 0 <= t <= 1
# u(0, x) = -sin(pi * x), u(t, -1) = u(t, 1) = 0

def make_tensor(x, requires_grad=True):
    t = torch.from_numpy(x)
    t.requires_grad=requires_grad
    t = t.float()
    return t

def make_training_initial_data(i_size, zero_shot=False, alpha=None, beta=None):
    
    x_i = np.random.uniform(low=-1.0, high=1.0, size=(i_size, 1))
    u_i = make_tensor(-np.sin(np.pi * x_i))
    t_i = make_tensor(np.zeros((i_size, 1)))
    x_i = make_tensor(x_i)

    if zero_shot:
        alpha_i = make_tensor(np.full((i_size, 1), alpha))
        beta_i = make_tensor(np.full((i_size, 1), beta))
        return [torch.cat([x_i, alpha_i, beta_i], axis=1), u_i]

    return [torch.cat([x_i], axis=1), u_i]

def make_training_boundary_data(b_size, zero_shot=False, alpha=None, beta=None):
    
    x_b = np.vstack(((-1) * np.ones((b_size // 2, 1)), np.ones((b_size - b_size // 2, 1))))
    
    u_b = make_tensor(np.vstack((
                                np.full((b_size // 2, 1), np.sin((-1) * alpha) + np.cos(beta) - 0.1),
                                np.full((b_size - b_size // 2, 1), np.sin(alpha) + np.cos(beta) + 0.1)
    )))

    x_b = make_tensor(x_b)

    if zero_shot:
        alpha_b = make_tensor(np.full((b_size, 1), alpha))
        beta_b = make_tensor(np.full((b_size, 1), beta))
        # return [torch.cat([x_b, t_b, z_b], axis=1), u_b]
        return [torch.cat([x_b, alpha_b, beta_b], axis=1), u_b]

    # return [torch.cat([x_b, t_b], axis=1), u_b]
    return [torch.cat([x_b], axis=1), u_b]

def make_training_domain_data(f_size, zero_shot=False, alpha=None, beta=None):
    x_f = make_tensor(np.random.uniform(low=-1.0, high=1.0, size=(f_size, 1)))
    # t_f = make_tensor(np.random.uniform(low=0.0, high=1.0, size=(f_size, 1)))
    u_f = make_tensor(np.zeros((f_size, 1)))

    if zero_shot:
        alpha_f = make_tensor(np.full((f_size, 1), alpha))
        beta_f = make_tensor(np.full((f_size, 1), beta))
        return [torch.cat([x_f, alpha_f, beta_f], axis=1), u_f]

    return [torch.cat([x_f], axis=1), u_f]

def generate_data(i_size, b_size, f_size, zero_shot=False, alpha=None, beta=None):
    i_set = make_training_initial_data(i_size, zero_shot, alpha, beta) if i_size > 0 else None
    b_set = make_training_boundary_data(b_size, zero_shot, alpha, beta)
    f_set = make_training_domain_data(f_size, zero_shot, alpha, beta)

    return i_set, b_set, f_set

def lhs(size, lb=-1, rb=1):
    """Latin Hypercube Sampling

       Args:
            size (int): shape of the sampled vector
            lb (float): left boundary
            rb (float): right boundary

       Returns:
            ret (list): sampled vector
    """
    res = [] if size > 1 else 0
    length = (rb - lb) / size
    for i in range(size):
        sub_lb = lb + length * i
        val = np.random.rand() * length + sub_lb
        if size > 1:
            res.append(val)
        else:
            res = val
    return res
        
def generate_task(size, ood=False):
    """Generate PINN tasks. Sample alpha and beta of tasks (only support)

       Args:
            size (int): number of tasks
            ood (boolean): whether out-of-distribution tasks be sampled or not 
    
       Returns:
            tasks (list): tasks (support)
    """
    tasks = []
    for _ in range(size):
        # alpha_lb, alpha_rb = -1.0, 1.0
        # beta_lb, beta_rb = -1.0, 1.0
        if ood:
            p = np.random.rand()
            alpha_qry_lb, alpha_qry_rb = (-1.5, -1.0) if p < 0.5 else (1.0, 1.5)
            p = np.random.rand()
            beta_qry_lb, beta_qry_rb = (-1.5, -1.0) if p < 0.5 else (1.0, 1.5)
        else:
            alpha_qry_lb, alpha_qry_rb = -1.0, 1.0
            beta_qry_lb, beta_qry_rb = -1.0, 1.0
            
        # alpha_sup = lhs(1, alpha_lb, alpha_rb)
        # beta_sup = lhs(1, beta_lb, beta_rb)
        alpha_qry = lhs(1, alpha_qry_lb, alpha_qry_rb)
        beta_qry = lhs(1, beta_qry_lb, beta_qry_rb)
        task = (alpha_qry, beta_qry)
        tasks.append(task)
    
    return tasks


