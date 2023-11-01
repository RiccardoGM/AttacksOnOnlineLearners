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
from Modules import EmpiricalGreedyAttacksPytorch as EGAP
from Parameters import ParametersPerceptronMNIST as Par



##############################################################
#                                                            #
#                        Parameters                          #
#                                                            #
##############################################################

# Model and activation
model_type = Par.model_type
activation = Par.activation

# MNIST parameters
class1 = Par.class1
label_class1 = Par.label_class1
class2 = Par.class2
label_class2 = Par.label_class2
degrees = Par.degrees

# Input data parameters
dim_input = Par.dim_input
batch_size = Par.batch_size

# Dynamics parameters
learning_rate = Par.learning_rate
beta = Par.beta
transient_th = Par.transient_th
momentum = Par.momentum

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
control_cost_pref_arr = Par.control_cost_pref_arr
fut_pref = Par.fut_pref
weight_future = Par.weight_future
opt_pref = Par.opt_pref
fut_pref_interval = Par.fut_pref_interval
fut_pref_min = Par.fut_pref_min
fut_pref_max = Par.fut_pref_max

# N past experiments
n_past_experiments = Par.n_past_experiments
n_runs_experiments = Par.n_runs_experiments

# Custom exp parameter (control_cost_pref)
custom_control = True
if custom_control:
    idx = int(sys.argv[1])-1 # import action cost index from stdin
    control_cost_pref = control_cost_pref_arr[idx]
    print('Custom control cost pre-factor:', control_cost_pref)

# Strings/paths
export_while_running = Par.export_while_running
export_path = Par.export_path
exp_description = Par.exp_description



##############################################################
#                                                            #
#                    MNIST preprocessing                     #
#                                                            #
##############################################################

dataset_original = datasets.MNIST('./data', train=True, download=True,
                                  transform=torchvision.transforms.Compose([
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

# Convert to arrays
dataset_original_array = dataset_original.data.numpy()
dataset_original_array_labels = dataset_original.targets.numpy()
X_original = dataset_original_array
original_shape = X_original.shape[1:]
X_0 = X_original.reshape((X_original.shape[0], -1))
y = dataset_original_array_labels

# Data loader
original_data_loader = torch.utils.data.DataLoader(dataset_original, batch_size=batch_size, shuffle=True)

# Reduce dimensionality, flatten and standardize
scaler1 = StandardScaler()
scaler2 = StandardScaler()
X_complete_0 = dataset_original_array
X_complete_0 = X_complete_0.reshape((X_complete_0.shape[0], -1))
X_complete_1 = scaler1.fit_transform(X_complete_0)
pca = PCA(n_components=dim_input)
X_complete_2 = pca.fit_transform(X_complete_1)
X_complete_3 = scaler2.fit_transform(X_complete_2)
X_complete = X_complete_3
X_1 = scaler1.fit_transform(X_0)
pca = PCA(n_components=dim_input)
X_2 = pca.fit_transform(X_1)
X_3 = scaler2.fit_transform(X_2)
X = X_3
X_reconstructed = scaler1.inverse_transform(pca.inverse_transform(scaler2.inverse_transform(X)))
X_tensor = torch.Tensor(X) # transform to torch tensor
y_tensor = torch.Tensor(y)
dataset_reduced = TensorDataset(X_tensor,y_tensor) # create your datset



##############################################################
#                                                            #
#                      Run experiment                        #
#                                                            #
##############################################################

dataset = dataset_reduced #dataset_original

attacker_on = True
export_while_running = True
use_cuda_if_available = False
print('\n')
print('Batch size:', batch_size)
print('Attacker on:', attacker_on)
print('Optimize fut. weight:', opt_pref)
print('Use cuda if available:', use_cuda_if_available)
print('\n')

for i in range(n_past_experiments, n_past_experiments + n_runs_experiments):
    run_idx = i+1
    print('run %d/%d'%(run_idx, n_runs_experiments))
    values = (100*control_cost_pref, batch_size, run_idx)
    exp_description_head = 'cprefdiv100#%d_batchsize#%d__run#%d_' % values
    experiment_description = exp_description_head + exp_description

    # Set future weight opt. parameters
    if Par.opt_pref:
        if run_idx==1:
            opt_pref = True
            fut_pref = Par.fut_pref
        else:
            opt_pref = False
            fut_pref = results['fut_pref']

    results = EGAP.labelleddata_exp_greedy(model_type=model_type,
                                           dataset=dataset,
                                           n_timesteps=n_timesteps,
                                           n_past_timesteps=n_timesteps_past,
                                           eta=learning_rate,
                                           dim_input=dim_input,
                                           a_min=a_min,
                                           a_max=a_max,
                                           n_gridpoints=n_gridpoints,
                                           beta=beta,
                                           control_cost_pref=control_cost_pref,
                                           transient_th=n_timesteps_transient_th,
                                           fut_pref=fut_pref,
                                           opt_pref=opt_pref,
                                           fut_pref_interval=fut_pref_interval,
                                           fut_pref_min=fut_pref_min,
                                           fut_pref_max=fut_pref_max,
                                           weight_future=weight_future,
                                           export_while_running=export_while_running,
                                           export_path=export_path,
                                           exp_description=experiment_description,
                                           export_results=True,
                                           use_cuda_if_available=use_cuda_if_available,
                                           momentum=momentum,
                                           teacher_smoothlabels=teacher_smoothlabels)
