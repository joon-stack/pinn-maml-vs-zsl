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

def train(epochs=1000, lr=0.1, i_size=500, b_size=500, f_size=1000, load=False, zero_shot=False, z_list=None):
    batch_count = 1


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    model = PINN(20, 3)

    if zero_shot:
        model = PINN(20, 5, zero_shot)

    if load:
        model.load_state_dict(torch.load('saved.data'))

    model.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    i_set = []
    b_set = []
    f_set = []

    if zero_shot:
        for z in z_list:
            i, b, f = generate_data(i_size, b_size, f_size, zero_shot, z)
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
        i_set, b_set, f_set = generate_data(i_size, b_size, f_size, zero_shot, z)
    

    dataset_train_initial  = CustomDataset(*i_set)
    dataset_train_boundary = CustomDataset(*b_set)
    dataset_train_domain   = CustomDataset(*f_set)

    zero_shot_num = len(z_list) if z_list is not None else 1

    loader_train_initial   = DataLoader(dataset_train_initial,  batch_size=i_size // batch_count * zero_shot_num)
    loader_train_boundary  = DataLoader(dataset_train_boundary, batch_size=b_size // batch_count * zero_shot_num)
    loader_train_domain    = DataLoader(dataset_train_domain,   batch_size=f_size // batch_count * zero_shot_num)   
    
    loss_func = nn.MSELoss()

    train_loss_i = []
    train_loss_b = []
    train_loss_f = []
    train_loss   = []

    loss_save = np.inf

    for epoch in range(epochs):
        model.train()
        
        for data_i, data_b, data_f in list(zip(loader_train_initial, loader_train_boundary, loader_train_domain)):
            optim.zero_grad()
        
            input_i, target_i = data_i
            input_b, target_b = data_b
            input_f, target_f = data_f
            

            input_i = input_i.to(device)
            target_i = target_i.to(device)
            input_b = input_b.to(device)
            target_b = target_b.to(device)
            input_f = input_f.to(device)
            target_f = target_f.to(device)
            
            loss_i = loss_func(target_i, model(input_i))
            loss_b = loss_func(target_b, model(input_b))
            loss_f = model.calc_loss_f(input_f, target_f)

            loss_i.to(device)
            loss_b.to(device)
            loss_f.to(device)

            loss = loss_i + loss_b * 10 + loss_f

            loss.backward()

            optim.step()

            train_loss_i += [loss_i.item()]
            train_loss_b += [loss_b.item()]
            train_loss_f += [loss_f.item()]
            train_loss   += [loss.item()]
        
        with torch.no_grad():
            model.eval()
            if loss_save > loss.item():
                torch.save(model.state_dict(), 'burgers.data')
                # print(".......model updated (epoch = ", epoch+1, ")")///
                loss_save = loss.item()

        if epoch % 100 == 99:
            with torch.no_grad():
                model.eval()
                print("Epoch {0} | Loss_I: {1:.4f} | Loss_B: {2:.4f} | Loss_F: {3:.4f}".format(epoch + 1, np.mean(train_loss_i), np.mean(train_loss_b), np.mean(train_loss_f)))
        
    return train_loss_i, train_loss_b, train_loss_f, train_loss, model



if __name__ == "__main__":
    train_loss_i, train_loss_b, train_loss_f, train_loss, model = train(zero_shot=True, z_list=np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]) / np.pi)

    