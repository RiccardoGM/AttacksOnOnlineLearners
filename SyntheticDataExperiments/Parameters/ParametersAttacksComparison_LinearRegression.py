# Import statements
import numpy as np
import torch

# Activation
activation = 'Linear'

# Input data parameters
dim_input = 10
batch_size = 1
mu_x = 0
sigma_x = 1
n_runs_experiments = 10

# Dynamics parameters
learning_rate = 2 * 1e-2 * dim_input
gamma = 0.995
beta = -np.log(gamma)/dim_input

# N. samples
n_timesteps = 5000
n_timesteps_transient_th = 20000 #20000
n_timesteps_past = 2*n_timesteps_transient_th
n_samples_average = 200
n_samples_buffer = 4*n_samples_average
n_samples_test = 10000
time_window = 1000

# Control parameters
a_min = -2 #-2, 0
a_max = 1-a_min #1
n_a_gridpoints = 101
n_runs_calibration = 10
control_cost_weight = 1.
greedy_weight_future = dim_input/learning_rate
opt_pref = True
fut_pref = 1

# DeepRL Agent
agent_model_name = 'TD3'
n_actions = 1
use_action_noise = True
action_noise_mean = np.zeros(n_actions)
action_noise_std = .2 * np.ones(n_actions)
use_small_achitecture = False
randomise_initial_condition = False
shuffle_array = True
learning_rate_agent = 0.0001
activation_fn = torch.nn.Tanh
n_episodes = 8
save_freq = 1000
train_freq = 100

# Strings/paths
local_path = '/Users/riccardo/Documents/GitHub/' #'path_to_progect_folder/'
export_path = local_path + 'OptimalControlAttacks/SyntheticDataExperiments/Results/StrategiesComparison/LinearRegression/'
values = (dim_input, batch_size, a_min, a_max, gamma*1000, 100*learning_rate, n_runs_experiments, n_runs_calibration)
experiment_description = '_dinput#%d_batchsize#%d_amin#%d_amax#%d_gamma1000#%d_lrpref100#%d_nav#%d_navopt#%d' % values
rlmodels_path = local_path + 'OptimalControlAttacks/SyntheticDataExperiments/RLAgents/LinearRegression/'
agent_model_fullname = 'Agent#%s__@@@_' % agent_model_name + experiment_description
agent_replaybuffer_fullname = 'RepBuffer#%s__@@@_' % agent_model_name + experiment_description
path_agent = rlmodels_path + agent_model_fullname
path_repbuffer = rlmodels_path +  agent_replaybuffer_fullname
