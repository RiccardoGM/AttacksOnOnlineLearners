# Import statements
import numpy as np


# Model and activation
model_type = 'LeNet'


# MNIST parameters
class1 = 1
label_class1 = 1
class2 = 7
label_class2 = -1
edge_length = 32
degrees = 10


# Input data parameters
batch_size = 1 #1, 10


# Dynamics parameters
learning_rate_LeNet = 1e-2
learning_rate = learning_rate_LeNet
gamma = 0.995
beta = -np.log(gamma)
transient_th = 25000 #25000
momentum = 0.
w_regularizer = 0


# Clean labels:
'''ground truth or teacher (model trained on clean data) '''
teacher_smoothlabels = True


# N. samples
n_timesteps_transient_th = transient_th
n_timesteps_past = int(2*n_timesteps_transient_th)
n_timesteps = int(2*n_timesteps_transient_th)
n_samples_average = 200
n_samples_buffer = 4*n_samples_average
n_samples_test = 1000


# Control parameters
a_min = -0
a_max = 1-a_min
n_gridpoints = 21
inv_lr = learning_rate**-1
#weight_future_arr = np.linspace(1, 19, 10)
weight_future_arr = np.linspace(inv_lr/10, inv_lr, 10)
#weight_future_arr = inv_lr + np.linspace(inv_lr/10, inv_lr, 10)
#weight_future_arr = np.linspace(inv_lr, inv_lr*10, 10)
idx = 4
control_cost_pref = 10**np.arange(-2., 2.5, .5)[idx] # idx=4 => C=1
fut_pref = 1


# N. averages
n_past_experiments = 0
n_runs_experiments = 10


# Strings/paths
export_while_running = True
local_path = '/Users/riccardo/Documents/GitHub/' #'path_to_progect_folder/'
export_path = local_path + 'OptimalControlAttacks/RealDataExperiments/Results/LeNetMNIST/'
values_1 = (a_min, a_max, n_gridpoints)
exp_description_1 = 'amin#%d_amax#%d_ngridpoints#%d' % values_1
values_2 = (n_timesteps_transient_th, gamma*1000, 100*learning_rate, 10*momentum)
exp_description_2 = exp_description_1 + '_nstepstransient#%d_gamma1000#%d_lrdiv100#%d_momentumdiv10#%d' % values_2
if w_regularizer>0:
    exp_description_2 = exp_description_2 + '_abslogwreg#%d' % -np.log10(w_regularizer)
if teacher_smoothlabels:
    exp_description_3 = exp_description_2 + '_teachsmoothing#True'
else:
    exp_description_3 = exp_description_2
if model_type=='LeNetLinLast':
    exp_description = exp_description_3 + '_LinOutput'
else:
    exp_description = exp_description_3
