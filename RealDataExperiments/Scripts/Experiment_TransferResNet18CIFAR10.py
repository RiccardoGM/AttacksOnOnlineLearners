##############################################################
#                                                            #
#                      Import libraries                      #
#                                                            #
##############################################################

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import copy

import numpy as np

import sys
import os
local_path = '/Users/riccardo/Documents/GitHub/' #'path_to_progect_folder/'
sys.path.append(local_path+'OptimalControlAttacks/RealDataExperiments/')
from Modules import GreedyAttacksPytorch as GAP



##############################################################
#                                                            #
#                        Parameters                          #
#                                                            #
##############################################################

# Activation
activation = 'Erf'

# Input data parameters
dim_input = 10
batch_size = 1
batch_size_list = [1, 10]
class1 = 3 # 3-cat, 4-deer
class2 = 5 # 5-dog, 7-horse

# Dynamics parameters
lr_pretraining = 1e-2
n_epochs_pretrain = 10
learning_rate = 2 * 1e-1 * dim_input
gamma = 0.995
beta = -np.log(gamma)/dim_input
w_regularizer = 2 * 1e-2
use_cuda = False

# N. samples
n_timesteps = 500
n_timesteps_transient_th = 250
n_timesteps_past = 2*n_timesteps_transient_th
n_samples_average = 200
n_samples_buffer = 4*n_samples_average
n_samples_test = 1000
n_samples_test_mini = 100

# Control parameters
a_min = -0 #0
a_max = 1-a_min #1
n_gridpoints = 21
control_cost_pref = 1.
control_cost_pref_arr = 10**np.linspace(-2., 2.4, 9)
fut_pref = 1.
future_weight = fut_pref / learning_rate

# N. averages
n_runs_experiments = 10
n_runs_calibration = 1

# Paths
path_data = local_path + 'OptimalControlAttacks/RealDataExperiments/ModelsData/CIFAR10/Classes_%d_%d/ResNet18/'%(class1, class2)
path_results = local_path + 'OptimalControlAttacks/RealDataExperiments/Results/TransferLearningCIFAR10/Classes_%d_%d/ResNet18/'%(class1, class2)



##############################################################
#                                                            #
#               ResNet CIFAR10 prelastfc data                #
#                                                            #
##############################################################

# Train data
filename = 'train_prelastfc_input.npy'
train_data_input = np.load(path_data+filename)
filename = 'train_prelastfc_label.npy'
train_data_label = np.load(path_data+filename)

# Test data
filename = 'test_prelastfc_input.npy'
test_data_input = np.load(path_data+filename)
filename = 'test_prelastfc_label.npy'
test_data_label = np.load(path_data+filename)

# Define train loader
tensor_x = torch.Tensor(train_data_input) # transform to torch tensor
tensor_y = torch.Tensor(train_data_label)
dataset = TensorDataset(tensor_x,tensor_y) # create your datset
dataloader_train = DataLoader(dataset, batch_size=batch_size) # create your dataloader
dataloader_pretrain = DataLoader(dataset, batch_size=100, shuffle=True) # create your dataloader
trainset = dataset

# Define test data
test_input = torch.Tensor(test_data_input) # transform to torch tensor
test_label = torch.Tensor(test_data_label)
dataset_test = TensorDataset(test_input, test_label)
testset = dataset_test



##############################################################
#                                                            #
#                Train teacher (MSE with L2)                 #
#                                                            #
##############################################################

# Set device
if use_cuda:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'

# Load parameters
filename = 'teacher_parameters.npy'
teacher_parameters_arr = np.load(path_data+filename)
teacher_parameters = torch.tensor(teacher_parameters_arr)
teacher_model = GAP.PerceptronModel(input_size=dim_input,
                                     activation=activation,
                                     parameters=teacher_parameters)
teacher_model_reg = copy.deepcopy(teacher_model).to(device)
teacher_model_reg.train()

# Define loss
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(teacher_model_reg.parameters(),
                      lr=lr_pretraining,
                      weight_decay=w_regularizer)

# Training with L2 loss
for epoch in range(n_epochs_pretrain):
    for idx, (x, label) in enumerate(dataloader_pretrain):
        x, label = x.to(device), label.to(device)
        label_pred = teacher_model_reg(x)
        loss = criterion(label_pred.flatten().double(), label.flatten().double())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

teacher_reg_parameters_arr = teacher_model_reg.linear1.weight.detach().cpu().numpy()
w_teach = teacher_reg_parameters_arr
w_target = -w_teach



##############################################################
#                                                            #
#                     Run experiments                        #
#                                                            #
##############################################################

results_dict = {}

for i, c_pref in enumerate(control_cost_pref_arr):
    results_dict['c#%d'%i] = {}
    for P in batch_size_list:
        print('\nc %.1e P %d' % (c_pref, P))

        # Initialize results dictionary
        results_dict['c#%d'%i]['P#%d'%P] = {}
        results_dict['c#%d'%i]['P#%d'%P]['w_dynamics'] = np.zeros((n_runs_experiments, n_timesteps, dim_input))
        results_dict['c#%d'%i]['P#%d'%P]['a_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['accuracy_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['d_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['nef_cost_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['per_cost_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['cum_cost_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['w_teach'] = np.zeros((n_runs_experiments, dim_input))
        results_dict['c#%d'%i]['P#%d'%P]['w_target'] = np.zeros((n_runs_experiments, dim_input))
        results_dict['c#%d'%i]['P#%d'%P]['fut_pref'] = np.ones(n_runs_experiments)
        results_dict['c#%d'%i]['P#%d'%P]['fut_pref_opt_grid'] = [0]*n_runs_experiments
        results_dict['c#%d'%i]['P#%d'%P]['running_cost_vs_fut_pref_opt_grid'] = [0]*n_runs_experiments

        # Set batch size
        batch_size = P

        #**************************************#
        #       Run multiple experiments       #
        #**************************************#

        for j, run in enumerate(range(n_runs_experiments)):
            if j%1==0:
                print('run %d/%d'%(run+1, n_runs_experiments))

            # Set future weight opt. parameters
            if run==0:
                opt_pref = True
                fut_pref = 1
            else:
                opt_pref = False
                fut_pref = results_dict['c#%d'%i]['P#%d'%P]['fut_pref'][0]
            fut_pref_min = 0.25
            fut_pref_interval = 0.25
            fut_pref_max = 10. + fut_pref_interval

            # Run experiment
            results_greedy = GAP.labelleddata_exp_greedy_ErfTL(trainset=trainset,
                                                                testset=testset,
                                                                w_teach=w_teach,
                                                                w_stud_0=w_teach,
                                                                n_timesteps=n_timesteps,
                                                                n_past_timesteps=n_timesteps_past,
                                                                eta=learning_rate,
                                                                dim_input=dim_input,
                                                                a_min=a_min,
                                                                a_max=a_max,
                                                                beta=beta,
                                                                control_cost_pref=c_pref,
                                                                batch_size=batch_size,
                                                                weight_future=future_weight,
                                                                n_gridpoints=n_gridpoints,
                                                                activation=activation,
                                                                opt_pref=opt_pref,
                                                                fut_pref_min=fut_pref_min,
                                                                fut_pref_max=fut_pref_max,
                                                                fut_pref_interval=fut_pref_interval,
                                                                fut_pref=fut_pref,
                                                                w_regularizer=w_regularizer)

            # Save results
            results_dict['c#%d'%i]['P#%d'%P]['w_dynamics'][run] = results_greedy['w_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['a_dynamics'][run] = results_greedy['a_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['accuracy_dynamics'][run] = results_greedy['accuracy_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['d_dynamics'][run] = results_greedy['d_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['nef_cost_dynamics'][run] = results_greedy['nef_cost_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['per_cost_dynamics'][run] = results_greedy['per_cost_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['cum_cost_dynamics'][run] = results_greedy['cum_cost_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['w_teach'][run] = w_teach
            results_dict['c#%d'%i]['P#%d'%P]['w_target'][run] = w_target
            results_dict['c#%d'%i]['P#%d'%P]['fut_pref'][run] = results_greedy['fut_pref']
            results_dict['c#%d'%i]['P#%d'%P]['fut_pref_opt_grid'][run] = results_greedy['fut_pref_opt_grid']
            results_dict['c#%d'%i]['P#%d'%P]['running_cost_vs_fut_pref_opt_grid'][run] = results_greedy['running_cost_vs_fut_pref_opt_grid']
            if run>0:
                results_dict['c#%d'%i]['P#%d'%P]['fut_pref_opt_grid'][run] = results_dict['c#%d'%i]['P#%d'%P]['fut_pref_opt_grid'][0]
                results_dict['c#%d'%i]['P#%d'%P]['running_cost_vs_fut_pref_opt_grid'][run] = results_dict['c#%d'%i]['P#%d'%P]['running_cost_vs_fut_pref_opt_grid'][0]



##############################################################
#                                                            #
#                     Export results                         #
#                                                            #
##############################################################

# Define string describing experiment
model_name = '_model#ResNet18CIFAR#%d#%d' % (class1, class2)
experiment_parameters_values = (dim_input, a_min, a_max, gamma*1000, 100*learning_rate/dim_input,
                                100*w_regularizer, n_runs_experiments, n_runs_calibration)
experiment_parameters = '_dinput#%d_amin#%d_amax#%d_gamma1000#%d_lrpref100#%d_wreg100#%d_nav#%d_navopt#%d' % experiment_parameters_values

# Export time_grid
name = 'time_grid'
filename = name + '_@@@' + model_name + experiment_parameters
duration = learning_rate * (n_timesteps-1)
timesteps = np.linspace(0, duration, n_timesteps)
data_to_export = timesteps
np.save(path_results + filename, data_to_export)

# Export c_pref_grid
name = 'c_pref_grid'
filename = name + '_@@@' + model_name + experiment_parameters
data_to_export = control_cost_pref_arr
np.save(path_results + filename, data_to_export)

# Export P, c_pref, dependent results
for idx_P, P in enumerate(batch_size_list):
    for i, c_pref in enumerate(control_cost_pref_arr):
        P_C_info = '_batchsize#%d_cprefidx#%d' % (P, i)
        for name in results_dict['c#%d'%i]['P#%d'%P].keys():
            filename = name + '_@@@' + model_name + P_C_info + experiment_parameters
            data_to_export = results_dict['c#%d'%i]['P#%d'%P][name]
            np.save(path_results + filename, data_to_export)
