# Import statements
import numpy as np


# Model and activation
model_type = 'Perceptron'
activation = 'Erf'


# MNIST parameters
class1 = 1
label_class1 = 1
class2 = 7
label_class2 = -1
degrees = 10


# Input data parameters
dim_input = 10
batch_size = 10


# Dynamics parameters
learning_rate_Perc = dim_input * 1e-2
learning_rate = learning_rate_Perc
gamma = 0.995
beta = -np.log(gamma)
transient_th = 10000 #10000
momentum = 0.


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
# take index as input
control_cost_pref = 1.
control_cost_pref_arr = 10**np.arange(-2., 2.5, .5)
weight_future = dim_input * learning_rate**-1
fut_pref = 1 #2.
fut_pref_arr = 10**np.linspace(-1, 1, 10)
opt_pref = True
calibrate_first_run_only = False
fut_pref_interval = .5
fut_pref_min = 1.
fut_pref_max = 5. + fut_pref_interval

# N. averages
n_past_experiments = 0
n_runs_experiments = 10


# Strings/paths
export_while_running = True
local_path = '/Users/riccardo/Documents/GitHub/' #'path_to_progect_folder/'
export_path = local_path + 'OptimalControlAttacks/RealDataExperiments/Results/PerceptronMNIST/'
values_1 = (dim_input, a_min, a_max, n_gridpoints)
exp_description_1 = '_dinput#%d_amin#%d_amax#%d_ngridpoints#%d' % values_1
values_2 = (n_timesteps_transient_th, gamma*1000, 100*learning_rate)
exp_description_2 = exp_description_1 + '_nstepstransient#%d_gamma1000#%d_lrdiv100#%d' % values_2
if teacher_smoothlabels:
    exp_description_3 = exp_description_2 + '_teachsmoothing#True'
else:
    exp_description_3 = exp_description_2
if opt_pref:
    exp_description = exp_description_3 + '_optpref#True'
else:
    exp_description = exp_description_3
