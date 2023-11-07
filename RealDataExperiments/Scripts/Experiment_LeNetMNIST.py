##############################################################
#                                                            #
#                      Import libraries                      #
#                                                            #
##############################################################

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import torch
import copy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
import os
local_path = '/Users/riccardo/Documents/GitHub/' #'path_to_progect_folder/'
sys.path.append(local_path+'OptimalControlAttacks/RealDataExperiments/')
from Modules import GreedyAttacksPytorch as GAP
from Parameters import ParametersLeNetMNIST as Par



##############################################################
#                                                            #
#                        Parameters                          #
#                                                            #
##############################################################

# Model and activation
model_type = Par.model_type

# MNIST parameters
class1 = Par.class1
label_class1 = Par.label_class1
class2 = Par.class2
label_class2 = Par.label_class2
edge_length = Par.edge_length
degrees = Par.degrees

# Input data parameters
batch_size = Par.batch_size

# Dynamics parameters
learning_rate = Par.learning_rate
gamma = Par.gamma
beta = Par.beta
transient_th = Par.transient_th
momentum = Par.momentum
w_regularizer = Par.w_regularizer

# Teacher or labels
teacher_smoothlabels = Par.teacher_smoothlabels

# N. samples
n_timesteps_transient_th = Par.n_timesteps_transient_th
n_timesteps_past = Par.n_timesteps_past
n_timesteps = Par.n_timesteps
n_samples_average = Par.n_samples_average
n_samples_buffer = Par.n_samples_buffer
n_samples_test = Par.n_samples_test

# Control parameters
a_min = Par.a_min
a_max = Par.a_max
n_gridpoints = Par.n_gridpoints
control_cost_pref = Par.control_cost_pref
weight_future_arr = Par.weight_future_arr
fut_pref = Par.fut_pref
n_runs_experiments = Par.n_runs_experiments
n_past_experiments = Par.n_past_experiments

# Custom exp parameters (c pref, run)
custom_w_fut = True
if custom_w_fut:
    idx = int(sys.argv[1])-1 # import future weight from stdin
    weight_fut = weight_future_arr[idx]
    print('Future weight:', weight_fut)

# Strings/paths
export_while_running = Par.export_while_running
export_path = Par.export_path



##############################################################
#                                                            #
#                    MNIST preprocessing                     #
#                                                            #
##############################################################

dataset_original = datasets.MNIST('./data', train=True, download=True,
                                  transform=torchvision.transforms.Compose([
                                      torchvision.transforms.Resize(edge_length),
                                      torchvision.transforms.RandomRotation(degrees),
                                      torchvision.transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))]))

# Mask classes
mask = (dataset_original.targets==class1) | (dataset_original.targets==class2)
dataset_original.targets = dataset_original.targets[mask]
dataset_original.data = dataset_original.data[mask]

# Reassign class values
mask = dataset_original.targets==class1
dataset_original.targets[mask] = label_class1
mask = dataset_original.targets==class2
dataset_original.targets[mask] = label_class2


##############################################################
#                                                            #
#                      Run experiment                        #
#                                                            #
##############################################################

dataset = dataset_original

attacker_on = True
use_cuda_if_available = True
opt_pref = False
print('\n')
print('Control cost pref:', control_cost_pref)
print('N gridpoints:', n_gridpoints)
print('Batch size:', batch_size)
print('Teacher labels:', teacher_smoothlabels)
print('Use cuda if available:', use_cuda_if_available)
print('\n')

for i in range(n_past_experiments, n_past_experiments + n_runs_experiments):
    run_idx = i+1
    print('run %d/%d'%(run_idx, n_runs_experiments))
    exp_description = Par.exp_description
    values = (100*control_cost_pref, batch_size, 100*weight_fut, run_idx)
    exp_description_head = 'cprefdiv100#%d_batchsize#%d_weightfutdiv100#%d_run#%d__@@__' % values
    exp_description = exp_description_head + exp_description

    GAP.labelleddata_exp_greedy(model_type=model_type,
                                 dataset=dataset,
                                 n_timesteps=n_timesteps,
                                 n_past_timesteps=n_timesteps_past,
                                 eta=learning_rate,
                                 batch_size=batch_size,
                                 a_min=a_min,
                                 a_max=a_max,
                                 n_gridpoints=n_gridpoints,
                                 beta=beta,
                                 control_cost_pref=control_cost_pref,
                                 transient_th=n_timesteps_transient_th,
                                 fut_pref=fut_pref,
                                 opt_pref=opt_pref,
                                 weight_future=weight_fut,
                                 export_while_running=export_while_running,
                                 export_path=export_path,
                                 exp_description=exp_description,
                                 attacker_on=attacker_on,
                                 use_cuda_if_available=use_cuda_if_available,
                                 export_results=True,
                                 momentum=momentum,
                                 w_regularizer=w_regularizer,
                                 teacher_smoothlabels=teacher_smoothlabels)
