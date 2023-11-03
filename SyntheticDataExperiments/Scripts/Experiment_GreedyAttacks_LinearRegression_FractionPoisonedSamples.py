##############################################################
#                                                            #
#                      Import libraries                      #
#                                                            #
##############################################################

import numpy as np
import scipy as sp

import sys
import os
local_path = '/Users/riccardo/Documents/GitHub/' #'path_to_progect_folder/'
sys.path.append(local_path+'OptimalControlAttacks/SyntheticDataExperiments/')
from Modules import EmpiricalGreedyAttacks as EGA
from Parameters import ParametersGreedyAttacks_LinearRegression as Par



##############################################################
#                                                            #
#                         Parameters                         #
#                                                            #
##############################################################

# Activation
activation = Par.activation

# Input data parameters
dim_input = Par.dim_input
mu_x = Par.mu_x
sigma_x = Par.sigma_x
batch_size_large = Par.batch_size_large

# Dynamics parameters
learning_rate = Par.learning_rate
gamma = Par.gamma
beta = Par.beta

# N. samples
n_timesteps = Par.n_timesteps
n_timesteps_transient_th = Par.n_timesteps_transient_th
n_timesteps_past = Par.n_timesteps_past
n_samples_average = Par.n_samples_average
n_samples_buffer = Par.n_samples_buffer
n_samples_test = Par.n_samples_test

# Control parameters
a_min = Par.a_min
a_max = Par.a_max
n_a_gridpoints = Par.n_a_gridpoints
greedy_weight_future_linear = Par.greedy_weight_future_linear
control_cost_weight = Par.control_cost_weight
control_cost_weight_arr = Par.control_cost_weight_arr
opt_pref = Par.opt_pref
fut_pref = Par.fut_pref
fraction_poisoned_samples_arr = Par.fraction_poisoned_samples_arr

# N. averages
n_runs_experiments = Par.n_runs_experiments
n_runs_calibration = Par.n_runs_calibration

# Test sets
x_test = np.random.normal(mu_x, sigma_x, (n_samples_test, dim_input))

# Strings/paths
export_path = Par.export_path
experiment_description = Par.experiment_description



##############################################################
#                                                            #
#                  Run multiple experiments                  #
#                                                            #
##############################################################

results_dict = {}

for i, c_pref in enumerate(control_cost_weight_arr):
    results_dict['c#%d'%i] = {}
    for fraction_poisoned in fraction_poisoned_samples_arr:
        print('rho: %.2f' % fraction_poisoned)

        # Initialize results dictionary
        results_dict['c#%d'%i]['fracpois100#%d'%(100*fraction_poisoned)] = {}
        results_dict['c#%d'%i]['fracpois100#%d'%(100*fraction_poisoned)]['d_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))

        for run in range(n_runs_experiments):
            print('run %d/%d'%(run+1, n_runs_experiments))

            # Teacher
            w_teach = np.random.normal(0, 1, dim_input)
            w_teach = w_teach/(np.sum(w_teach**2)/dim_input)**0.5

            # Target
            w_target = -w_teach

            # Student (initial condition)
            w_stud_0 = w_teach

            # Weight control
            d_teach_target = 0.5 * np.mean((EGA.perceptron(w_teach, x_test, activation=activation)-EGA.perceptron(w_target, x_test, activation=activation))**2)
            control_cost_weight_run = c_pref

            # Arrays (assuming batch size as specified by 'batch_size')
            x_incoming = np.random.normal(mu_x, sigma_x, (batch_size_large*n_timesteps, dim_input))
            x_past = np.random.normal(mu_x, sigma_x, (batch_size_large*n_timesteps_past, dim_input))
            x_buffer = np.random.normal(mu_x, sigma_x, (batch_size_large*n_samples_buffer, dim_input))

            # Future weight opt. parameters
            if Par.opt_pref:
                if run==0:
                    opt_pref = True
                    fut_pref = Par.opt_pref
                else:
                    opt_pref = False
                    fut_pref = results_dict['c#%d'%i]['P#%d'%P]['fut_pref'][0]
            fut_pref_interval = 0.1
            fut_pref_min = 0.1
            fut_pref_max = 10. + fut_pref_interval

            # Run single experiment
            results_greedy = EGA.exp_greedy_perceptron(x_incoming=x_incoming,
                                                       x_past=x_past,
                                                       x_buffer=x_buffer,
                                                       x_test=x_test,
                                                       dim_input=dim_input,
                                                       w_teach=w_teach,
                                                       w_target=w_target,
                                                       w_stud_0=w_stud_0,
                                                       eta=learning_rate,
                                                       beta=beta,
                                                       control_cost_weight=control_cost_weight_run,
                                                       a_min=a_min,
                                                       a_max=a_max,
                                                       batch_size=batch_size_large,
                                                       weight_future=greedy_weight_future_linear,
                                                       buffer_size=n_samples_average,
                                                       activation=activation,
                                                       transient_th=n_timesteps_transient_th,
                                                       fut_pref=fut_pref,
                                                       opt_pref=opt_pref,
                                                       fut_pref_interval=fut_pref_interval,
                                                       fut_pref_min=fut_pref_min,
                                                       fut_pref_max=fut_pref_max,
                                                       n_av=n_runs_calibration,
                                                       n_gridpoints=n_a_gridpoints,
                                                       fraction_poisoned_samples=fraction_poisoned)

            # Save results
            results_dict['c#%d'%i]['fracpois100#%d'%(100*fraction_poisoned)]['d_dynamics'][run] = results_greedy['d_dynamics']


##############################################################
#                                                            #
#                       Export results                       #
#                                                            #
##############################################################

experiment_description = '_batchsize#%d' % batch_size_large + experiment_description

# Export c_pref_grid
name = 'c_pref_grid'
filename = name + '__@@@_' + experiment_description
data_to_export = control_cost_weight_arr
np.save(export_path + filename, data_to_export)

# Export P, c_pref, dependent results
for fraction_poisoned in fraction_poisoned_samples_arr:
    for i, c_pref in enumerate(control_cost_weight_arr):
        fracpois_C_info = '_fracpois100#%d_cprefidx#%d' % (100*fraction_poisoned, i)
        for name in results_dict['c#%d'%i]['fracpois100#%d'%(100*fraction_poisoned)].keys():
            filename = name + '__@@@_' + fracpois_C_info + '__@@_' + experiment_description
            data_to_export = results_dict['c#%d'%i]['fracpois100#%d'%(100*fraction_poisoned)][name]
            np.save(export_path + filename, data_to_export)
