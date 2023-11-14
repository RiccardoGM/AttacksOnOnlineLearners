## Import statements ##

import numpy as np

# Stable baselines
import gymnasium as gym

# Custom modules
import sys
import os
local_path = '/Users/riccardo/Documents/GitHub/' #'path_to_progect_folder/'
sys.path.append(local_path+'OptimalControlAttacks/SyntheticDataExperiments/')
from Modules import GreedyAttacks as GA



# ************************************************ #
#                                                  #
#              Poisoning Environment               #
#                                                  #
# ************************************************ #

# ***************** Perceptron ******************** #

class EnvironmentPerceptron(gym.Env):

    def __init__(self, x_arr, batch_size, w_stud_0, w_teach, w_target, a_min,
                 a_max, learning_rate, control_cost_weight, activation='Erf',
                 randomise_initial_condition=False, shuffle_array=True):

        super(EnvironmentPerceptron, self).__init__()

        # Input data to self
        self.dim_input = x_arr.shape[1]
        self.batch_size = batch_size
        self.n_timesteps = int(x_arr.shape[0]/batch_size)
        self.learning_rate = learning_rate
        self.control_cost_weight = control_cost_weight
        self.activation = activation
        self.w_teach = w_teach
        self.w_target = w_target

        # Set time
        self.timestep = 0

        # Initialize the student
        self.randomise_initial_condition = randomise_initial_condition
        self.rho_grid = np.linspace(0.0, 1.0, 7)
        self.rho_idx = 0
        if randomise_initial_condition:
            np.random.shuffle(self.rho_grid)
            rho = self.rho_grid[self.rho_idx]
            self.w_stud_0 = self.w_teach * (1-rho) + self.w_target * rho
        else:
            self.w_stud_0 = w_stud_0
        self.w_stud = self.w_stud_0

        # Data stream
        self.shuffle_array = shuffle_array
        if shuffle_array:
            np.random.shuffle(x_arr)
        self.x_arr = x_arr.astype(np.float32)
        self.x_batch = self.x_arr[self.timestep*self.batch_size:(self.timestep+1)*self.batch_size,:]
        self.x_batch_next = self.x_arr[(self.timestep+1)*self.batch_size:(self.timestep+2)*self.batch_size,:]

        # Define action and observation space
        self.a_min = a_min
        self.a_max = a_max
        self.action_space = gym.spaces.Box(low=self.a_min, high=self.a_max, shape=(1,), dtype=np.float32)
        high = 1e9
        self.observation_space = gym.spaces.Box(low=-high, high=high, shape=((1+self.batch_size)*self.dim_input,), dtype=np.float32)


    def reset(self, seed=0):
        # Re-initialise the student
        if self.randomise_initial_condition:
            if self.rho_idx<len(self.rho_grid)-1:
                self.rho_idx += 1
                rho = self.rho_grid[self.rho_idx]
            else:
                np.random.shuffle(self.rho_grid)
                self.rho_idx = 0
                rho = self.rho_grid[self.rho_idx]
            self.w_stud_0 = self.w_teach * (1-rho) + self.w_target * rho
        self.w_stud = self.w_stud_0
        # Shuffle array
        if self.shuffle_array:
            np.random.shuffle(self.x_arr)
        # Reset time
        self.timestep = 0
        # Reset batches
        self.x_batch = self.x_arr[self.timestep*self.batch_size:(self.timestep+1)*self.batch_size,:]
        self.x_batch_next = self.x_arr[(self.timestep+1)*self.batch_size:(self.timestep+2)*self.batch_size,:]
        return_info = {'Rand. initial condition': self.randomise_initial_condition,
                       'Shuffle input data': self.shuffle_array}
        return_obs = np.concatenate((self.w_stud.reshape(1,-1), self.x_batch), axis=0, dtype=np.float32).flatten()
        return return_obs, return_info


    def student_update(self, a):
        w_stud_next = GA.student_update_perceptron(w_stud=self.w_stud,
                                                   w_teach=self.w_teach,
                                                   w_target=self.w_target,
                                                   x_batch=self.x_batch,
                                                   a=a,
                                                   eta=self.learning_rate,
                                                   dim_input=self.dim_input,
                                                   activation=self.activation)
        return w_stud_next


    def step(self, action):

        # Perturbation cost
        per_cost = self.control_cost_weight * 0.5 * action**2

        # Nefarious cost
        input_s = np.dot(self.w_stud, self.x_batch.T)/(self.dim_input**0.5)
        input_o = np.dot(self.w_target, self.x_batch.T)/(self.dim_input**0.5)
        activation_s = GA.perceptron(input_s, activation=self.activation)
        activation_o = GA.perceptron(input_o, activation=self.activation)
        nef_cost = 0.5 * np.mean((activation_s-activation_o)**2)

        # Update
        self.w_stud = self.student_update(action)
        self.timestep += 1
        self.x_batch = self.x_arr[self.timestep*self.batch_size:(self.timestep+1)*self.batch_size,:]
        self.x_batch_next = self.x_arr[(self.timestep+1)*self.batch_size:(self.timestep+2)*self.batch_size,:]

        # Total cost and reward
        total_cost = per_cost + nef_cost
        reward = float(-total_cost)

        # End episode flag
        terminated = bool(self.timestep > self.n_timesteps-2)
        truncated = False

        # Info
        info = {'Action': action,
                'Per. cost': per_cost,
                'Nef cost': nef_cost}

        # Observation
        return_obs = np.concatenate((self.w_stud.reshape(1,-1), self.x_batch), axis=0, dtype=np.float32).flatten()

        return return_obs, reward, terminated, truncated, info


# ************* 2-layer NN (NN2L) ***************** #

class EnvironmentNN2L(gym.Env):

    def __init__(self, observation_type, x_arr, batch_size, W_teach, v_teach, W_target, v_target,
                 W_stud_0, v_stud_0, a_min, a_max, learning_rate, control_cost_weight, activation='Erf',
                 outputscaling='inv_sqroot', hlsize=100, shuffle_array=True, train_first_layer=True):

        super(EnvironmentNN2L, self).__init__()

        # Input data to self
        self.dim_input = x_arr.shape[1]
        self.batch_size = batch_size
        self.n_timesteps = int(x_arr.shape[0]/batch_size)
        self.learning_rate = learning_rate #.astype(np.float32)
        self.control_cost_weight = control_cost_weight #.astype(np.float32)
        self.activation = activation
        self.outputscaling = outputscaling
        self.train_first_layer = train_first_layer
        self.hlsize = hlsize
        self.W_teach = W_teach.astype(np.float32)
        self.v_teach = v_teach.astype(np.float32)
        self.W_target = W_target.astype(np.float32)
        self.v_target = v_target.astype(np.float32)

        # Set time
        self.timestep = 0

        # Initialize the student
        self.W_stud_0 = W_stud_0
        self.v_stud_0 = v_stud_0
        self.W_stud = self.W_stud_0
        self.v_stud = self.v_stud_0

        # Data stream
        self.shuffle_array = shuffle_array
        if shuffle_array:
            np.random.shuffle(x_arr)
        self.x_arr = x_arr.astype(np.float32)
        self.x_batch = self.x_arr[self.timestep*self.batch_size:(self.timestep+1)*self.batch_size,:]
        self.x_batch_next = self.x_arr[(self.timestep+1)*self.batch_size:(self.timestep+2)*self.batch_size,:]

        # Define action and observation space
        self.a_min = a_min
        self.a_max = a_max
        self.action_space = gym.spaces.Box(low=self.a_min, high=self.a_max, shape=(1,), dtype=np.float32)
        self.observation_type = observation_type
        high = 1e9
        if self.observation_type=='full':
            self.observation_space = gym.spaces.Box(low=-high, high=high,
                                                    shape=(self.dim_input*(1+self.hlsize)+self.hlsize,),
                                                    dtype=np.float32)
        elif self.observation_type=='lastlayer':
            self.observation_space = gym.spaces.Box(low=-high, high=high,
                                                    shape=(self.dim_input+self.hlsize,),
                                                    dtype=np.float32)
        elif self.observation_type=='input':
            self.observation_space = gym.spaces.Box(low=-high, high=high,
                                                    shape=(self.dim_input,),
                                                    dtype=np.float32)
        else:
            raise ValueError('observation must be one of the following: full, lastlayer, input.')


    def reset(self, seed=0):
        # Re-initialise the student
        self.W_stud = self.W_stud_0
        self.v_stud = self.v_stud_0
        # Shuffle array
        if self.shuffle_array:
            np.random.shuffle(self.x_arr)
        # Reset time
        self.timestep = 0
        # Reset batches
        self.x_batch = self.x_arr[self.timestep*self.batch_size:(self.timestep+1)*self.batch_size,:]
        self.x_batch_next = self.x_arr[(self.timestep+1)*self.batch_size:(self.timestep+2)*self.batch_size,:]
        return_info = {'Shuffle input data': self.shuffle_array}

        if self.observation_type=='full':
            return_obs = np.concatenate((self.v_stud.reshape(1,-1),
                                         self.W_stud.reshape(1,-1),
                                         self.x_batch.reshape(1,-1)), axis=1,
                                         dtype=np.float32).flatten()
        elif self.observation_type=='lastlayer':
            return_obs = np.concatenate((self.v_stud.reshape(1,-1),
                                         self.x_batch.reshape(1,-1)), axis=1,
                                         dtype=np.float32).flatten()
        elif self.observation_type=='input':
            return_obs = self.x_batch.flatten()
        return return_obs, return_info


    def student_update(self, a):

        W_stud_next, v_stud_next = GA.student_update_NN2L(W_stud=self.W_stud,
                                                          v_stud=self.v_stud,
                                                          W_target=self.W_target,
                                                          v_target=self.v_target,
                                                          W_teach=self.W_teach,
                                                          v_teach=self.v_teach,
                                                          x_batch=self.x_batch,
                                                          a=a,
                                                          eta=self.learning_rate,
                                                          dim_input=self.dim_input,
                                                          activation=self.activation,
                                                          output_scaling=self.outputscaling,
                                                          train_first_layer=self.train_first_layer)
        return W_stud_next[0], v_stud_next[0]


    def step(self, action):

        # Perturbation cost
        per_cost = self.control_cost_weight * 0.5 * action**2

        # Nefarious cost
        label_o = GA.NN2L(self.x_batch,
                          self.W_target,
                          self.v_target,
                          activation=self.activation,
                          output_scaling=self.outputscaling)
        label_s = GA.NN2L(self.x_batch,
                          self.W_stud,
                          self.v_stud,
                          activation=self.activation,
                          output_scaling=self.outputscaling)
        nef_cost = 0.5 * np.mean((label_s-label_o)**2)

        # Update
        self.W_stud, self.v_stud = self.student_update(action)
        self.timestep += 1
        self.x_batch = self.x_arr[self.timestep*self.batch_size:(self.timestep+1)*self.batch_size,:]
        self.x_batch_next = self.x_arr[(self.timestep+1)*self.batch_size:(self.timestep+2)*self.batch_size,:]

        # Total cost and reward
        total_cost = per_cost + nef_cost
        reward = float(-total_cost)

        # End episode flag
        terminated = bool(self.timestep > self.n_timesteps-2)
        truncated = False

        # Info
        info = {'Action': action,
                'Per. cost': per_cost,
                'Nef cost': nef_cost}

        # Observation
        if self.observation_type=='full':
            return_obs = np.concatenate((self.v_stud.reshape(1,-1),
                                         self.W_stud.reshape(1,-1),
                                         self.x_batch.reshape(1,-1)), axis=1,
                                         dtype=np.float32).flatten()
        elif self.observation_type=='lastlayer':
            return_obs = np.concatenate((self.v_stud.reshape(1,-1),
                                         self.x_batch.reshape(1,-1)), axis=1,
                                         dtype=np.float32).flatten()
        elif self.observation_type=='input':
            return_obs = self.x_batch.flatten()

        return return_obs, reward, terminated, truncated, info


# ************************************************* #
#                                                   #
#                    Experiment                     #
#                                                   #
# ************************************************* #

# ***************** Perceptron ******************** #

def exp_rl_perceptron(model, env, x_test):

    # Parameters from env
    dim_input = env.dim_input
    batch_size = env.batch_size
    n_timesteps = env.n_timesteps
    learning_rate = env.learning_rate
    control_cost_weight = env.control_cost_weight
    activation = env.activation
    w_teach = env.w_teach
    w_target = env.w_target

    # Target distance and target labels
    n_samples_test = x_test.shape[0]
    input_t_test = np.dot(w_teach, x_test.T)/(dim_input**0.5)
    input_o_test = np.dot(w_target, x_test.T)/(dim_input**0.5)
    label_t_test = GA.perceptron(input_t_test, activation=activation)
    label_o_test = GA.perceptron(input_o_test, activation=activation)
    error_target_teach = np.mean((label_o_test-label_t_test)**2)
    d_target_teach = 0.5 * error_target_teach
    t_pred_test = np.sign(label_t_test)

    # Arrays to save dynamics
    w_stud_dynamics = np.zeros((n_timesteps-1, dim_input))
    a_dynamics = np.zeros(n_timesteps-1)
    nef_cost_dynamics = np.zeros(n_timesteps-1)
    per_cost_dynamics = np.zeros(n_timesteps-1)
    cum_cost_dynamics = np.zeros(n_timesteps-1)
    d_dynamics = np.zeros(n_timesteps-1)
    acc_dynamics = np.zeros(n_timesteps-1)

    # Reset environment
    observation, _ = env.reset()

    # Run experiment
    for step in range(n_timesteps):
        # State, distance, accuracy
        w_stud = env.w_stud
        x_batch = env.x_batch
        w_stud_dynamics[step] = w_stud
        input_s_test = np.dot(w_stud, x_test.T)/(dim_input**0.5)
        label_s_test = GA.perceptron(input_s_test, activation=activation)
        s_pred_test = np.sign(label_s_test)
        d_student_teach = 0.5 * np.mean((label_s_test-label_t_test)**2)
        d_dynamics[step] = (d_student_teach/d_target_teach)**0.5
        acc_dynamics[step] = np.sum(s_pred_test==t_pred_test.reshape(1,-1), axis=1)/n_samples_test

        # Nefarious cost
        input_s = np.dot(w_stud, x_batch.T)/(dim_input**0.5)
        input_o = np.dot(w_target, x_batch.T)/(dim_input**0.5)
        label_s = GA.perceptron(input_s, activation=activation)
        label_o = GA.perceptron(input_o, activation=activation)
        nef_cost_dynamics[step] = 0.5 * np.mean((label_s-label_o)**2)

        # Step
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # Action and costs
        a_dynamics[step] = action
        per_cost_dynamics[step] = 0.5 * control_cost_weight * action**2
        cum_cost_dynamics[step] = per_cost_dynamics[step] + nef_cost_dynamics[step]

        # End of episode
        if terminated==True:
            break

    # Results dictionary
    results = {}
    results['w_dynamics'] = w_stud_dynamics
    results['a_dynamics'] = a_dynamics
    results['nef_cost_dynamics'] = nef_cost_dynamics
    results['per_cost_dynamics'] = per_cost_dynamics
    results['cum_cost_dynamics'] = cum_cost_dynamics
    results['d_dynamics'] = d_dynamics
    results['accuracy_dynamics'] = acc_dynamics

    return results


# ************* 2-layer NN (NN2L) ***************** #

def exp_rl_NN2L(model, env, x_test):

    # Parameters from env
    dim_input = env.dim_input
    batch_size = env.batch_size
    n_timesteps = env.n_timesteps
    learning_rate = env.learning_rate
    control_cost_weight = env.control_cost_weight
    activation = env.activation
    outputscaling = env.outputscaling
    train_first_layer = env.train_first_layer
    hlsize = env.hlsize
    W_teach = env.W_teach
    v_teach = env.v_teach
    W_target = env.W_target
    v_target = env.v_target

    # Target distance and target labels
    n_samples_test = x_test.shape[0]
    label_t_test = GA.NN2L(x_test, W_teach, v_teach, activation=activation, output_scaling=outputscaling)
    label_o_test = GA.NN2L(x_test, W_target, v_target, activation=activation, output_scaling=outputscaling)
    error_target_teach = np.mean((label_o_test-label_t_test)**2)
    d_target_teach = 0.5 * error_target_teach
    pref_control_cost_weight = (2*d_target_teach)
    t_pred_test = np.sign(label_t_test)

    # Arrays to save dynamics
    a_dynamics = np.zeros(n_timesteps-1)
    nef_cost_dynamics = np.zeros(n_timesteps-1)
    per_cost_dynamics = np.zeros(n_timesteps-1)
    cum_cost_dynamics = np.zeros(n_timesteps-1)
    d_dynamics = np.zeros(n_timesteps-1)
    acc_dynamics = np.zeros(n_timesteps-1)

    # Reset environment
    observation, _ = env.reset()

    # Run experiment
    for step in range(n_timesteps):
        # State, distance, accuracy
        W_stud = env.W_stud
        v_stud = env.v_stud
        x_batch = env.x_batch
        label_s_test = GA.NN2L(x_test, W_stud, v_stud, activation=activation, output_scaling=outputscaling)
        s_pred_test = np.sign(label_s_test)
        d_student_teach = 0.5 * np.mean((label_s_test-label_t_test)**2)
        d_dynamics[step] = (d_student_teach/d_target_teach)**0.5
        acc_dynamics[step] = np.sum(s_pred_test==t_pred_test.reshape(1,-1), axis=1)/n_samples_test

        # Nefarious cost
        label_s = GA.NN2L(x_batch, W_stud, v_stud, activation=activation, output_scaling=outputscaling)
        label_o = GA.NN2L(x_batch, W_target, v_target, activation=activation, output_scaling=outputscaling)
        nef_cost_dynamics[step] = 0.5 * np.mean((label_s-label_o)**2)

        # Step
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # Action and costs
        a_dynamics[step] = action
        per_cost_dynamics[step] = 0.5 * control_cost_weight * action**2
        cum_cost_dynamics[step] = per_cost_dynamics[step] + nef_cost_dynamics[step]

        # End of episode
        if terminated==True:
            break

    # Results dictionary
    results = {}
    results['a_dynamics'] = a_dynamics
    results['nef_cost_dynamics'] = nef_cost_dynamics
    results['per_cost_dynamics'] = per_cost_dynamics
    results['cum_cost_dynamics'] = cum_cost_dynamics
    results['d_dynamics'] = d_dynamics
    results['accuracy_dynamics'] = acc_dynamics

    return results
