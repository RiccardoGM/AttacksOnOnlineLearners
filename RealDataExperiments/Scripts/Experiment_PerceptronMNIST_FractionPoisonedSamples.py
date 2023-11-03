##############################################################
#                                                            #
#                      Import libraries                      #
#                                                            #
##############################################################

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import torch

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
batch_size = Par.batch_size_large

# Dynamics parameters
learning_rate = Par.learning_rate
gamma = Par.gamma
beta = Par.beta
transient_th = Par.transient_th
momentum = Par.momentum

# Teacher or labels
teacher_smoothlabels = Par.teacher_smoothlabels # True in manuscript

# N. samples
n_timesteps_transient_th = Par.n_timesteps_transient_th
n_timesteps_past = Par.n_timesteps_past
n_timesteps = Par.n_timesteps
n_samples_average = Par.n_samples_average
n_samples_buffer = Par.n_samples_buffer
n_samples_test = Par.n_samples_test

# Control parameters
a_min = Par.a_min # -2 in manuscript
a_max = Par.a_max # 3 in manuscript
n_gridpoints = Par.n_gridpoints
control_cost_pref = Par.control_cost_pref # 1 in manuscript
fut_pref = Par.fut_pref
n_runs_experiments = Par.n_runs_experiments
n_past_experiments = Par.n_past_experiments
weight_fut = dim_input/learning_rate
fraction_poisoned_samples_arr = Par.fraction_poisoned_samples_arr

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
#                  Run standard experiment                   #
#                                                            #
##############################################################

''' Run experiment with fraction_poisoned_samples=1 to calibrate fut_pref '''

dataset = dataset_reduced
fraction_poisoned_samples = 1.
attacker_on = True
use_cuda_if_available = False
opt_pref = False #True (set to false as we found fut_pref - can re-run)
fut_pref = 1.6
fut_pref_interval = .125
fut_pref_min = 0.75
fut_pref_max = 2.5

print('\n')
print('Fut. pref:', fut_pref)
print('Control cost pref:', control_cost_pref)
print('N gridpoints:', n_gridpoints)
print('Batch size:', batch_size)
print('Teacher labels:', teacher_smoothlabels)
print('Use cuda if available:', use_cuda_if_available)
print('\n')

res = EGAP.labelleddata_exp_greedy(model_type=model_type,
                                   activation=activation,
                                   dataset=dataset,
                                   dim_input=dim_input,
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
                                   fut_pref_interval=fut_pref_interval,
                                   fut_pref_min=fut_pref_min,
                                   fut_pref_max=fut_pref_max,
                                   opt_pref=opt_pref,
                                   weight_future=weight_fut,
                                   export_while_running=False,
                                   export_path=export_path,
                                   exp_description=exp_description,
                                   attacker_on=attacker_on,
                                   use_cuda_if_available=use_cuda_if_available,
                                   export_results=False,
                                   momentum=momentum,
                                   teacher_smoothlabels=teacher_smoothlabels,
                                   fraction_poisoned_samples=fraction_poisoned_samples)



##############################################################
#                                                            #
#                       Run experiment                       #
#                                                            #
##############################################################

''' Run experiment with variable fraction_poisoned_samples '''

fut_pref = res['fut_pref']
opt_pref = False
results_dict = {}

for fraction_poisoned in fraction_poisoned_samples_arr:
    print('rho: %.2f' % fraction_poisoned)

    # Initialize results dictionary
    results_dict['fracpois100#%d'%(100*fraction_poisoned)] = {}
    results_dict['fracpois100#%d'%(100*fraction_poisoned)]['d_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))

    for run in range(n_runs_experiments):
        print('run %d/%d'%(run+1, n_runs_experiments))

        res_fracpois = EGAP.labelleddata_exp_greedy(model_type=model_type,
                                                    activation=activation,
                                                    dataset=dataset,
                                                    dim_input=dim_input,
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
                                                    fut_pref_interval=fut_pref_interval,
                                                    fut_pref_min=fut_pref_min,
                                                    fut_pref_max=fut_pref_max,
                                                    opt_pref=opt_pref,
                                                    weight_future=weight_fut,
                                                    export_while_running=False,
                                                    export_path=export_path,
                                                    exp_description=exp_description,
                                                    attacker_on=attacker_on,
                                                    use_cuda_if_available=use_cuda_if_available,
                                                    export_results=False,
                                                    momentum=momentum,
                                                    teacher_smoothlabels=True,
                                                    fraction_poisoned_samples=fraction_poisoned)

        # Save result
        results_dict['fracpois100#%d'%(100*fraction_poisoned)]['d_dynamics'][run] = res_fracpois['d_dynamics']



##############################################################
#                                                            #
#                       Export results                       #
#                                                            #
##############################################################

experiment_description = '_batchsize#%d_' % batch_size + exp_description

# Export P, c_pref, dependent results
for fraction_poisoned in fraction_poisoned_samples_arr:
    fracpois_C_info = '_fracpois100#%d_cpref100#%d' % (100*fraction_poisoned, 100*control_cost_pref)
    for name in results_dict['fracpois100#%d'%(100*fraction_poisoned)].keys():
        filename = name + '__@@@_' + fracpois_C_info + '__@@_' + experiment_description
        data_to_export = results_dict['fracpois100#%d'%(100*fraction_poisoned)][name]
        np.save(export_path + filename, data_to_export)
