import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt


from model import *
from generate_data import *
from utils import *

from copy import copy

class CustomDataset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target
    
    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]
        return input, target



def train(epochs=1000, lr=0.1, i_size=500, b_size=500, f_size=1000, load=False, zero_shot=False, alpha_list=None, beta_list=None, load_data=None, low=-1, high=1, eqname='poisson'):
    batch_count = 1
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    if eqname == 'poisson':
        model = PINN(20, 5, zero_shot, dim=1, param_num=2)

    elif eqname == 'burgers':
        model = PINN(20, 8, zero_shot, dim=2, param_num=1)

    if load:
        model.load_state_dict(torch.load(load_data))
        print("Model loaded succefully {}".format(load_data))

    model = model.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # optim = torch.optim.LBFGS(model.parameters(),
    #                           max_iter=50000,
    #                           max_eval=50000,
    #                           )

    i_set = []
    b_set = []
    f_set = []

    if eqname == 'poisson':
        if zero_shot:
            for alpha, beta in zip(alpha_list, beta_list):
                i, b, f = generate_data(i_size, b_size, f_size, zero_shot, alpha, beta, low, high)
                if i:
                    i_set.append(i)
                b_set.append(b)
                f_set.append(f)
            
            i_set = list(zip(*i_set))
            for n, i in enumerate(i_set):
                i_set[n] = torch.cat(i, axis=0)
            
            b_set = list(zip(*b_set))
            for n, b in enumerate(b_set):
                b_set[n] = torch.cat(b, axis=0)

            f_set = list(zip(*f_set))
            for n, f in enumerate(f_set):
                f_set[n] = torch.cat(f, axis=0)
        else:
            alpha = alpha_list
            beta = beta_list
            i_set, b_set, f_set = generate_data(i_size, b_size, f_size, zero_shot, alpha, beta, low, high)

    elif eqname == 'burgers':
        if zero_shot:
            for alpha in alpha_list:
                i, b, f = generate_data_burgers(i_size, b_size, f_size, zero_shot, alpha, low, high)
                i_set.append(i)
                b_set.append(b)
                f_set.append(f)

            i_set = list(zip(*i_set))
            for n, i in enumerate(i_set):
                i_set[n] = torch.cat(i, axis=0)
            
            b_set = list(zip(*b_set))
            for n, b in enumerate(b_set):
                b_set[n] = torch.cat(b, axis=0)

            f_set = list(zip(*f_set))
            for n, f in enumerate(f_set):
                f_set[n] = torch.cat(f, axis=0)
        else:
            alpha = alpha_list
            i_set, b_set, f_set = generate_data_burgers(i_size, b_size, f_size, zero_shot, alpha, low, high)
    
    if eqname == 'poisson':
        # zero_shot_num = len(alpha_list) * len(beta_list) if zero_shot else 1
        zero_shot_num = len(alpha_list) if zero_shot else 1
    elif eqname == 'burgers':
        zero_shot_num = len(alpha_list) if zero_shot else 1

    if i_size > 0:
        dataset_train_initial  = CustomDataset(*i_set)
        loader_train_initial   = DataLoader(dataset_train_initial, num_workers=0, batch_size=i_size // batch_count * zero_shot_num)
        train_loss_i = []

    dataset_train_boundary = CustomDataset(*b_set)
    dataset_train_domain   = CustomDataset(*f_set)

    loader_train_boundary  = DataLoader(dataset_train_boundary, num_workers=0, batch_size=b_size // batch_count * zero_shot_num)
    loader_train_domain    = DataLoader(dataset_train_domain,   num_workers=0, batch_size=f_size // batch_count * zero_shot_num)   
    
    print("Data generation completed")

    loss_func = nn.MSELoss()

    train_loss_i = []
    train_loss_b = []
    train_loss_f = []
    train_loss   = []

    loss_save = np.inf

    val_loss = []
    val_ood_loss = []

    nrmse_set = []

    if eqname == 'burgers':
        for epoch in range(epochs):
            model.train()
            
            for data_i, data_b, data_f in list(zip(loader_train_initial, loader_train_boundary, loader_train_domain)):
                
            
                input_i, target_i = data_i
                input_b, target_b = data_b
                input_f, target_f = data_f

                input_i = input_i.to(device)
                target_i = target_i.to(device)
                input_b = input_b.to(device)
                target_b = target_b.to(device)
                input_f = input_f.to(device)
                target_f = target_f.to(device)

                optim.zero_grad()
                loss_i = loss_func(target_i, model(input_i))
                loss_b = loss_func(target_b, model(input_b))         
                loss_f = model.calc_loss_f_burgers(input_f, target_f) if zero_shot else model.calc_loss_f_burgers(input_f, target_f, alpha=alpha)
                loss = loss_i + loss_b + loss_f
                train_loss_i += [loss_i.item()]
                train_loss_b += [loss_b.item()]
                train_loss_f += [loss_f.item()]
                train_loss   += [loss.item()]
                loss.backward()
                optim.step()
                
            
            if epoch % 100 == 99:
                with torch.no_grad():
                    model.eval()
                    print("Epoch {0} | Loss_I: {1:.4f} | Loss_B: {2:.4f} | Loss_F: {3:.4f}".format(epoch + 1, np.mean(train_loss_i), np.mean(train_loss_b), np.mean(train_loss_f)))
            
            if epoch % 100 == 99:
                with torch.no_grad():
                    model.eval()
                    alpha_val = [0.01 / np.pi]
                    alpha_val_ood = [0.2 / np.pi]
                    x = np.linspace(-1, 1, num=101).reshape(-1, 1)
                    t = np.linspace(0, 1, num=101).reshape(-1, 1)
                    x, t = np.meshgrid(x, t)
                    x = x.reshape(-1, 1)
                    t = t.reshape(-1, 1)
                    
                    loss_val = evaluate_burgers(x, t, alpha_val, model, device, zero_shot)
                    loss_val_ood = evaluate_burgers(x, t, alpha_val_ood, model, device, zero_shot)

                    val_loss += [loss_val]
                    val_ood_loss += [loss_val_ood]

                    print("alpha: {}, Val. NRMSE: {:.3f} ".format(alpha_val, loss_val.item()))
                    print("alpha: {}, Val. OOD NRMSE: {:.3f} ".format(alpha_val_ood, loss_val_ood.item()))
            
            if epoch % 1000 == 999:
                with torch.no_grad():
                    torch.save(model.state_dict(), 'models/burgers_zs_{}.data'.format(epoch + 1))

    elif eqname == 'poisson':
        for epoch in range(epochs):
            model.train()
            
            for data_b, data_f in list(zip(loader_train_boundary, loader_train_domain)):
                optim.zero_grad()
            
                input_b, target_b = data_b
                input_f, target_f = data_f

                input_b = input_b.to(device)
                target_b = target_b.to(device)
                input_f = input_f.to(device)
                target_f = target_f.to(device)

                # print("Input",input_b[0])
                # print("Pred", model(input_b)[0][0])
                # print("Target", target_b[0][0])
                loss_b = loss_func(target_b, model(input_b))
                loss_f = 0

                for m in range(zero_shot_num):
                    input_f_batch = input_f[f_size * m : f_size * (m + 1)]
                    target_f_batch = target_f[f_size * m : f_size * (m + 1)]
                    if zero_shot:
                        loss_f += model.calc_loss_f(input_f_batch, target_f_batch)
                    else:
                        loss_f += model.calc_loss_f(input_f_batch, target_f_batch, alpha, beta)
                
                loss_f /= zero_shot_num

                loss_b.to(device)
                loss_f.to(device)

                loss = loss_b * 10 + loss_f
                # print("{:.3f}".format(loss.item()))

                loss.backward()

                optim.step()

                train_loss_b += [loss_b.item()]
                train_loss_f += [loss_f.item()]
                train_loss   += [loss.item()]
            

            if epoch % 50 == 49:
                if zero_shot:
                    with torch.no_grad():
                        model.eval()
                        alpha_val, beta_val = sample_data(200, -1, 1), sample_data(200, -1, 1)
                        alpha_val_ood, beta_val_ood = sample_data(100, 1, 1.5), sample_data(100, 1, 1.5)
                        alpha_val_ood.extend(sample_data(100, -1.5, -1))
                        beta_val_ood.extend(sample_data(100, -1.5, -1))
                        x = np.linspace(low, high, num=100).reshape(-1, 1)
                        
                        loss_val = evaluate(x, alpha_val, beta_val, model, device)
                        loss_val_ood = evaluate(x, alpha_val_ood, beta_val_ood, model, device)

                        val_loss += [loss_val]
                        val_ood_loss += [loss_val_ood]

                        print("alpha, beta: {:.3f}, {:.3f} Val. NRMSE: {:.3f} | alpha, beta: {:.3f}, {:.3f} Val. OOD NRMSE: {:.3f}".format(alpha_val[0], beta_val[0], loss_val.item(), alpha_val_ood[0], beta_val_ood[0], loss_val_ood.item()))
                
                    

            if epoch % 100 == 99:

                with torch.no_grad():
                    model.eval()
                    print("Epoch {0} | Loss_I: {1:.4f} | Loss_B: {2:.4f} | Loss_F: {3:.4f}".format(epoch + 1, np.mean(train_loss_i), np.mean(train_loss_b), np.mean(train_loss_f)))
                    if not zero_shot:
                        nrmse = model.validate(alpha=alpha_list, beta=beta_list, low=low, high=high)
                        print("NRMSE: {:.4f}".format(nrmse))
                        nrmse_set.append(nrmse)
            
            if epoch % 1000 == 999:
                with torch.no_grad():
                    fname = 'models/poisson_zs_{}.data'.format(epoch + 1) if zero_shot else 'models/poisson_{}.data'
                    torch.save(model.state_dict(), fname)
            

        
    return train_loss_i, train_loss_b, train_loss_f, train_loss, model, val_loss, val_ood_loss, nrmse_set


if __name__ == "__main__":
    # train_loss_i, train_loss_b, train_loss_f, train_loss, model = train(zero_shot=True, z_list=np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]) / np.pi)
    # alpha = np.random.rand() * 2 - 1
    # beta = np.random.rand() * 2 - 1
    # print("Alpha: {:.3f}, Beta: {:.3f}".format(alpha, beta))
    # train_loss_i, train_loss_b, train_loss_f, train_loss, model = train(zero_shot=False, alpha_list=alpha, beta_list=beta)
    alpha = np.random.uniform(low=-1, high=1, size=1000)
    beta = np.random.uniform(low=-1, high=1, size=1000)
    loss_i, loss_b, loss_f, loss, model, val_loss, val_ood_loss, nrmse = train(epochs=1000, 
                                                                                lr=0.01, 
                                                                                i_size=0, 
                                                                                b_size=2, 
                                                                                f_size=1, 
                                                                                zero_shot=True, 
                                                                                # alpha_list=np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]) / np.pi, 
                                                                                alpha_list=alpha,
                                                                                beta_list=beta,
                                                                                low=-1, 
                                                                                high=1, 
                                                                                eqname='poisson')

    