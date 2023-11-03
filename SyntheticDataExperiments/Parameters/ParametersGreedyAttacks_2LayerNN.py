# Import statements
import numpy as np

# Activation
activation = 'Erf'

# Target type
target_type = 'FlippedTeacher' #'Random'

# Input data parameters
dim_input = 10
mu_x = 0
sigma_x = 1
batch_size = 1
batch_size_list = [1, 10]
hiddenlayer_width = 100
output_scaling = 'inv_sqroot' # 'inv_sqroot' or 'inv'

# Dynamics parameters
learning_rate = 2 * 1e-2 * dim_input * hiddenlayer_width**0.5
gamma = 0.995
beta = -np.log(gamma)/dim_input

# N. samples
n_timesteps = 5000 #5000
n_timesteps_transient_th = 2500 #2500
n_timesteps_past = 2*n_timesteps_transient_th
n_samples_average = 500
n_samples_buffer = 4*n_samples_average
n_samples_test = 500

# Control parameters
a_min = -0
a_max = 1-a_min
n_a_gridpoints = 21
weight_future = dim_input/learning_rate
control_cost_weight = 1. #1.
control_cost_weight_arr = 10**np.arange(-2., 2.6, 0.4)
opt_pref = False
fut_pref = 2.

# N. averages
n_runs_experiments = 10 #10
n_runs_calibration = 4 #4

# Strings/paths
local_path = '/Users/riccardo/Documents/GitHub/' #'path_to_progect_folder/'
export_path = local_path + 'OptimalControlAttacks/SyntheticDataExperiments/Results/GreedyAttacks/2LayerNN/'
values = (dim_input, hiddenlayer_width, a_min, a_max, gamma*1000, 10000*learning_rate/dim_input, n_runs_experiments, n_runs_calibration)
experiment_description = '_dinput#%d_hlayer#%d_amin#%d_amax#%d_gamma1000#%d_lrpref10000#%d_nav#%d_navopt#%d' % values
if not opt_pref:
    experiment_description = experiment_description + '_optpref#False'
