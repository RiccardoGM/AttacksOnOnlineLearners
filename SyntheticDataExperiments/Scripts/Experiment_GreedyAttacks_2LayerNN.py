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
from Modules import GreedyAttacks as GA
from Parameters import ParametersGreedyAttacks_2LayerNN as Par



##############################################################
#                                                            #
#                        Parameters                          #
#                                                            #
##############################################################

# Model
activation = Par.activation
output_scaling = Par.output_scaling
hiddenlayer_width = Par.hiddenlayer_width
target_type = Par.target_type

# Input data parameters
dim_input = Par.dim_input
mu_x = Par.mu_x
sigma_x = Par.sigma_x
batch_size = Par.batch_size
batch_size_list = Par.batch_size_list

# Dynamics parameters
learning_rate = Par.learning_rate
gamma = Par.gamma
beta = Par.beta

# N. samples
n_timesteps_transient_th = Par.n_timesteps_transient_th
n_timesteps_past = Par.n_timesteps_past
n_timesteps = Par.n_timesteps
n_samples_average = Par.n_samples_average
n_samples_buffer = Par.n_samples_buffer
n_samples_test = Par.n_samples_test

# Test sets
x_test = np.random.normal(mu_x, sigma_x, (n_samples_test, dim_input))

# Control parameters
a_min = Par.a_min
a_max = Par.a_max
n_a_gridpoints = Par.n_a_gridpoints

# Control cost array
control_cost_weight_arr = Par.control_cost_weight_arr
control_cost_weight = Par.control_cost_weight

# Future weight
weight_future = Par.weight_future
fut_pref = Par.fut_pref
opt_pref = Par.opt_pref

# N. averages
n_runs_experiments = Par.n_runs_experiments
n_runs_calibration = Par.n_runs_calibration

# Strings/paths
export_path = Par.export_path
experiment_description = Par.experiment_description


##############################################################
#                                                            #
#                      Run experiment                        #
#                                                            #
##############################################################

attacker_on = True
print('\n')
print('Batch size:', batch_size)
print('Attacker on:', attacker_on)
print('Optimize fut. weight:', opt_pref)
print('\n')

results_dict = {}

for i, c_pref in enumerate(control_cost_weight_arr):
    results_dict['c#%d'%i] = {}
    for P in batch_size_list:
        print('c %.1e P %d' % (c_pref, P))

        # Initialize results dictionary
        results_dict['c#%d'%i]['P#%d'%P] = {}
        results_dict['c#%d'%i]['P#%d'%P]['a_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['d_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['accuracy_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['nef_cost_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['per_cost_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['cum_cost_dynamics'] = np.zeros((n_runs_experiments, n_timesteps))
        results_dict['c#%d'%i]['P#%d'%P]['fut_pref'] = np.ones(n_runs_experiments)
        results_dict['c#%d'%i]['P#%d'%P]['fut_pref_opt_grid'] = [0]*n_runs_experiments
        results_dict['c#%d'%i]['P#%d'%P]['running_cost_vs_fut_pref_opt_grid'] = [0]*n_runs_experiments

        # Set batch size
        batch_size = P

        # Set cost of action
        control_cost_weight_run = c_pref

        for run in range(n_runs_experiments):
            print('run %d/%d'%(run+1, n_runs_experiments))

            # Teacher
            W_teach = np.random.normal(0, 1, (hiddenlayer_width, dim_input))
            W_teach = W_teach / (np.sum(W_teach**2, axis=1).reshape(-1,1).repeat(dim_input, axis=1)/dim_input)**0.5
            v_teach = np.random.normal(0, 1, hiddenlayer_width)
            v_teach = v_teach / ((np.sum(v_teach**2)/hiddenlayer_width))**0.5

            # Target
            if target_type=='FlippedTeacher':
                W_target = W_teach.copy()
                v_target = -v_teach.copy()
            elif target_type=='Random':
                W_target = np.random.normal(0, 1, (hiddenlayer_width, dim_input))
                W_target = W_target / (np.sum(W_target**2, axis=1).reshape(-1,1).repeat(dim_input, axis=1)/dim_input)**0.5
                v_target = np.random.normal(0, 1, hiddenlayer_width)
                v_target = v_target / ((np.sum(v_target**2)/hiddenlayer_width))**0.5

            # Student (initial condition)
            W_stud_0 = W_teach.copy()
            v_stud_0 = v_teach.copy()

            # Arrays (assuming batch_size as specified above)
            x_incoming = np.random.normal(mu_x, sigma_x, (batch_size*n_timesteps, dim_input))
            x_past = np.random.normal(mu_x, sigma_x, (batch_size*n_timesteps_past, dim_input))
            x_buffer = np.random.normal(mu_x, sigma_x, (batch_size*n_samples_buffer, dim_input))

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
            results_greedy = GA.exp_greedy_NN2L(x_incoming=x_incoming,
                                                 x_past=x_past,
                                                 x_buffer=x_buffer,
                                                 x_test=x_test,
                                                 W_teach=W_teach,
                                                 v_teach=v_teach,
                                                 W_target=W_target,
                                                 v_target=v_target,
                                                 W_stud_0=W_stud_0,
                                                 v_stud_0=v_stud_0,
                                                 control_cost_weight=control_cost_weight_run,
                                                 eta=learning_rate,
                                                 dim_input=dim_input,
                                                 beta=beta,
                                                 a_min=a_min,
                                                 a_max=a_max,
                                                 batch_size=batch_size,
                                                 weight_future=weight_future,
                                                 buffer_size=n_samples_average,
                                                 activation=activation,
                                                 transient_th=n_timesteps_transient_th,
                                                 fut_pref=fut_pref,
                                                 opt_pref=opt_pref,
                                                 n_av=n_runs_calibration,
                                                 n_gridpoints=n_a_gridpoints,
                                                 output_scaling=output_scaling)

            # Save results
            results_dict['c#%d'%i]['P#%d'%P]['a_dynamics'][run] = results_greedy['a_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['d_dynamics'][run] = results_greedy['d_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['accuracy_dynamics'][run] = results_greedy['accuracy_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['nef_cost_dynamics'][run] = results_greedy['nef_cost_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['per_cost_dynamics'][run] = results_greedy['per_cost_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['cum_cost_dynamics'][run] = results_greedy['cum_cost_dynamics']
            results_dict['c#%d'%i]['P#%d'%P]['fut_pref'][run] = results_greedy['fut_pref']
            results_dict['c#%d'%i]['P#%d'%P]['fut_pref_opt_grid'][run] = results_greedy['fut_pref_opt_grid']
            results_dict['c#%d'%i]['P#%d'%P]['running_cost_vs_fut_pref_opt_grid'][run] = results_greedy['running_cost_vs_fut_pref_opt_grid']



##############################################################
#                                                            #
#                       Export results                       #
#                                                            #
##############################################################

# Export c_pref_grid
name = 'c_pref_grid'
filename = name + '__@@@_' + experiment_description
data_to_export = control_cost_weight_arr
np.save(export_path + filename, data_to_export)

# Export P, c_pref, dependent results
for idx_P, P in enumerate(batch_size_list):
    for i, c_pref in enumerate(control_cost_weight_arr):
        P_C_info = '_batchsize#%d_cprefidx#%d' % (P, i)
        for name in results_dict['c#%d'%i]['P#%d'%P].keys():
            filename = name + '__@@@_' + P_C_info + '__@@_' + experiment_description
            data_to_export = results_dict['c#%d'%i]['P#%d'%P][name]
            np.save(export_path + filename, data_to_export)
