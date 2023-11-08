'''
    This file contains implementations of clairvoyant label attacks,
    for linear regression, perceptron, and two-layer fully connected networks.

    All of these functions take as input one array of data:

    1) x_incoming (batch_size * n_timesteps, dim_input): the future data
       used to perform attacks and that are known to the attacker.
       The current implementation assumes batch_size=1.

    Other inputs include: parameters of teacher and target models, parameters
    of dynamics (learning rate, etc.), parameters of attacks (boundaries
    for actions, etc.).
'''

# Import statements
import numpy as np
import scipy.special as spsc
import ipopt
import sympy as sym
from collections import OrderedDict
from opty.direct_collocation import Problem
from opty.utils import building_docs



# ************************************************* #
#                                                   #
#                Auxiliary functions                #
#                                                   #
# ************************************************* #

# ***************** Soft ReLU ********************* #

''' softReLU and derivative (step function) '''

s_SReLU = 0.2

def SReLU(x, s=s_SReLU):
    return x * 0.5 * spsc.erfc(-(x/s)/np.sqrt(2)) + s * np.exp(-(x/s)**2/2.)/np.sqrt(2*np.pi)

def dSReLU(x, s=s_SReLU):
    return 0.5 * spsc.erfc(-(x/s)/np.sqrt(2))


# ******************* ReLU *********************** #

''' ReLU and derivative (step function) '''

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

# ************** Error function ****************** #


''' Erf and derivative '''

def Erf(x):
    return spsc.erf(x)

def dErf(x):
    return (2./np.sqrt(np.pi)) * np.exp(-x**2)



# ************************************************* #
#                                                   #
#                      Models                       #
#                                                   #
# ************************************************* #

# ***************** Perceptron ******************** #

''' Symbolic perceptron '''

def perceptron_sym(x, s=s_SReLU, activation='Erf'):

    if activation=='Erf':
        output = sym.erf(x/(2**0.5))
    elif activation=='SReLU':
        output = 0.5*x*sym.erfc(-(x/s_SReLU)/(2**0.5)) + s_SReLU*sym.exp(-(x/s_SReLU)**2/2)/ np.sqrt(2*np.pi)
    elif activation=='ReLU':
        raise ValueError('ReLU not supported')
    elif activation=='Linear':
        output = x
    else:
        raise ValueError('Activation not uderstood.')

    return output


''' Numerical perceptron '''

def perceptron(x, s=s_SReLU, activation='Erf'):

    if activation=='Erf':
        output = Erf(x/(2**0.5))
    elif activation=='SReLU':
        output = SReLU(x, s=s)
    elif activation=='ReLU':
        raise ValueError('ReLU not supported')
    elif activation=='Linear':
        output = x
    else:
        raise ValueError('Activation not uderstood.')

    return output


''' Derivative numerical perceptron '''

def d_perceptron(x, s=s_SReLU, activation='Erf'):

    if activation=='Erf':
        output = (1/np.sqrt(2)) * dErf(x/np.sqrt(2))
    elif activation=='SReLU':
        output = dSReLU(x, s=s)
    elif activation=='ReLU':
        raise ValueError('ReLU not supported')
    elif activation=='Linear':
        output = np.ones_like(x)
    else:
        raise ValueError('Activation not uderstood.')

    return output


# ************* 2-layer NN (NN2L) ***************** #

''' 2-layer NN '''

def NN2L(X, W, v, dim_input, hlayer_size, s=s_SReLU, activation='Erf', output_scaling='inv_sqroot'):

    if len(X.shape)<2:
        X = np.expand_dims(X, axis=0)
    batch_size = X.shape[0]
    if len(W.shape)<3:
        W = W.reshape(1,*W.shape)
    if len(v.shape)<2:
        v = v.reshape(1,*v.shape)
    v = v.reshape(*v.shape, 1)

    X = np.atleast_3d(X)
    input_L1 = np.matmul(W, X)/np.sqrt(dim_input)

    # Activation
    if activation=='Erf':
        activation_L1 = Erf(input_L1/(2**0.5))
    elif activation=='SReLU':
        activation_L1 = SReLU(input_L1)
    elif activation=='ReLU':
        raise ValueError('ReLU not supported')
    elif activation=='Linear':
        activation_L1 = input_L1
    else:
        raise ValueError('Activation not uderstood.')

    # Pre-factor
    if output_scaling=='inv':
        pref_hlayer_size = hlayer_size
    elif output_scaling=='inv_sqroot':
        pref_hlayer_size = np.sqrt(hlayer_size)
    else:
        raise ValueError('output_scaling not understood')

    output = np.sum(v*activation_L1, axis=1)/pref_hlayer_size

    return np.squeeze(output)



# ************************************************* #
#                                                   #
#                    Experiment                     #
#                                                   #
# ************************************************* #

# ***************** Perceptron ******************** #

''' NB: Opty requires at least two state variables: dim_input>=2 '''

def exp_clairv_perceptron(x_incoming, x_test, eta, w_teach, w_target, dim_input,
                          w_stud_0=[], control_cost_weight=1., a_min=0, a_max=1,
                          beta=0.001, activation='Erf'):

    # N state variables
    n_states = dim_input

    # Time grid
    n_timesteps = x_incoming.shape[0]
    duration = eta * (n_timesteps-1)
    timesteps = np.linspace(0, duration, n_timesteps) # t = 0, 1*eta, ...n_timesteps*eta

    # Control cost weight
    control_cost_pref = control_cost_weight
    input_t_test = np.dot(w_teach, x_test.T)/(dim_input**0.5)
    input_o_test = np.dot(w_target, x_test.T)/(dim_input**0.5)
    label_t_test = perceptron(input_t_test, activation=activation)
    label_o_test = perceptron(input_o_test, activation=activation)
    error_target_teach = np.mean((label_o_test-label_t_test)**2)
    d_target_teach = 0.5 * error_target_teach
    control_cost_weight = control_cost_pref * error_target_teach

    # Symbols
    t = sym.symbols('t')
    x = sym.symbols('x0:%d' % (dim_input), cls=sym.Function)
    w_teach_sym = sym.symbols('w_t0:%d' % (dim_input))
    w_stud_sym = sym.symbols('w_s0:%d' % (dim_input), cls=sym.Function)
    w_target_sym = sym.symbols('w_st0:%d' % (dim_input))
    a = sym.symbols('a', cls=sym.Function)

    # Symbols grouped
    state_symbols = tuple([w_stud_sym[i](t) for i in range(dim_input)])
    constant_symbols_1 = [w_teach_sym[i] for i in range(dim_input)]
    constant_symbols_2 = [w_target_sym[i] for i in range(dim_input)]
    constant_symbols = tuple(constant_symbols_1 + constant_symbols_2)
    specified_symbols = (a(t), ) # useful?

    # Trajectory map for x_incoming
    trajectory_map = {}
    for dim in range(dim_input):
        trajectory_map[x[dim](t)] = x_incoming[:, dim]

    # Input
    input_s = 0
    for i in range(dim_input):
        input_s += w_stud_sym[i](t)*x[i](t)/dim_input**0.5
    input_t = 0
    for i in range(dim_input):
        input_t += w_teach_sym[i]*x[i](t)/dim_input**0.5
    input_o = 0
    for i in range(dim_input):
        input_o += w_target_sym[i]*x[i](t)/dim_input**0.5

    # Activation
    if activation=='Erf':
        activation_s = sym.erf(input_s/(2**0.5))
        activation_t = sym.erf(input_t/(2**0.5))
        activation_o = sym.erf(input_o/(2**0.5))
    elif activation=='SReLU':
        activation_s = 0.5*input_s*sym.erfc(-(input_s/s_SReLU)/(2**0.5)) + s_SReLU*sym.exp(-(input_s/s_SReLU)**2/2)/ np.sqrt(2*np.pi)
        activation_t = 0.5*input_t*sym.erfc(-(input_t/s_SReLU)/(2**0.5)) + s_SReLU*sym.exp(-(input_t/s_SReLU)**2/2)/ np.sqrt(2*np.pi)
        activation_o = 0.5*input_o*sym.erfc(-(input_o/s_SReLU)/(2**0.5)) + s_SReLU*sym.exp(-(input_o/s_SReLU)**2/2)/ np.sqrt(2*np.pi)
    elif activation=='ReLU':
        raise ValueError('ReLU not supported')
        activation_s = 0.5*(sym.sign(input_s)+1)*input_s  #sym.Piecewise((0,input_s<=0),(input_s, input_s>0))
        activation_t = 0.5*(sym.sign(input_t)+1)*input_t  #sym.Piecewise((0,input_t<=0),(input_t, input_t>0))
        activation_o = 0.5*(sym.sign(input_o)+1)*input_o  #sym.Piecewise((0,input_o<=0),(input_o, input_o>0))
    elif activation=='Linear':
        activation_s = input_s
        activation_t = input_t
        activation_o = input_o
    else:
        raise ValueError('Activation must be one of the following: Erf, SReLU, Linear')
    activation_dagger = activation_t*(1-a(t)) + activation_o*a(t)

    # Loss function
    Loss_fct = (activation_s - activation_dagger)**2 / 2.

    # Loss function grad
    Loss_fct_diff = tuple([sym.diff(Loss_fct, w_stud_sym[i](t)) for i in range(dim_input)])

    # Equations of motion
    eom_list = [w_stud_sym[i](t).diff() + Loss_fct_diff[i] for i in range(dim_input)]
    eom = sym.Matrix(eom_list)

    # Specify the known system parameters
    par_map = OrderedDict()
    for i in range(dim_input):
        par_map[w_teach_sym[i]] = w_teach[i]
        par_map[w_target_sym[i]] = w_target[i]

    # Specify the objective function and it's gradient
    def obj(free):
        '''
           Objective function to be minimized: discounted sum of action and nefarious costs.
           Takes as input the free vector, which stores student parameters and actions at all times:

           -> free = [state_0_t0, ...state_0_tend, state_1_t0, ...state_1_tend, ...state_last_tend,
                      control_t0, ...control_tend]
                      state_i_tj = w_stud_i_tj : student parameters are "state variables"
                      control_tj = a_tj : action is "control variable"
              so the first n_states * n_timesteps elements in free are state variables at all times,
              followed by n_timesteps elements that are the control variable at all times.

              nef_cost_tj = 0.5 * np.dot(state_tj-w_target, x_incoming[j])**2 / dim_input
              per_cost_tj = 0.5 * control_cost_weight * control_tj**2
              total_cost = Sum_j (per_cost_tj+nef_cost_tj) * eta * exp(-beta * eta * j)
        '''

        Cost = Cost_control = Cost_nef = 0

        # Cost due to state variables
        state_variables = free[0:n_states*n_timesteps].reshape((n_states, n_timesteps)).T
        input_s_l = np.einsum('ij,ji->i', state_variables, x_incoming.T)/dim_input**0.5
        input_o_l = np.dot(w_target, x_incoming.T)/dim_input**0.5
        if activation=='Erf':
            activation_s_l = Erf(input_s_l/(2**0.5))
            activation_o_l = Erf(input_o_l/(2**0.5))
        elif activation=='ReLU':
            activation_s_l = ReLU(input_s_l)
            activation_o_l = ReLU(input_o_l)
        elif activation=='SReLU':
            activation_s_l = SReLU(input_s_l)
            activation_o_l = SReLU(input_o_l)
        elif activation=='Linear':
            activation_s_l = input_s_l
            activation_o_l = input_o_l
        Cost_nef = 0.5*(activation_s_l-activation_o_l)**2
        Cost_nef = np.sum(eta * Cost_nef * np.exp(-beta*timesteps))

        # Cost due to control variables
        control_variables = free[n_states*n_timesteps:]
        Cost_control = 0.5 * control_cost_weight * control_variables**2
        Cost_control = np.sum(eta * Cost_control * np.exp(-beta*timesteps))

        return Cost_nef + Cost_control


    def obj_grad(free):
        '''
           Gradient of the objective function with respect to the free vector.
        '''

        grad = np.zeros_like(free)

        # Grad w.r.t. state variable
        state_variables = free[0:n_states*n_timesteps].reshape((n_states, n_timesteps)).T
        input_s_l = np.einsum('ij,ji->i', state_variables, x_incoming.T)/dim_input**0.5
        input_o_l = np.dot(w_target, x_incoming.T)/dim_input**0.5
        if activation=='Erf':
            activation_s_l = Erf(input_s_l/(2**0.5))
            activation_o_l = Erf(input_o_l/(2**0.5))
            d_activation_s = (1/np.sqrt(2*dim_input)) * dErf(input_s_l/(2**0.5))
        elif activation=='ReLU':
            activation_s_l = ReLU(input_s_l)
            activation_o_l = ReLU(input_o_l)
            d_activation_s = (1/np.sqrt(dim_input)) * dReLU(input_s_l)
        elif activation=='SReLU':
            activation_s_l = SReLU(input_s_l)
            activation_o_l = SReLU(input_o_l)
            d_activation_s = (1/np.sqrt(dim_input)) * dSReLU(input_s_l)
        elif activation=='Linear':
            activation_s_l = input_s_l
            activation_o_l = input_o_l
            d_activation_s = (1/np.sqrt(dim_input))
        grad_nef_timefactors = (activation_s_l-activation_o_l) * d_activation_s
        grad_nef_timefactors = eta * grad_nef_timefactors * np.exp(-beta*timesteps)
        for state in range(n_states):
            grad[state*n_timesteps:(state+1)*n_timesteps] = grad_nef_timefactors * x_incoming[:, state]

        # Grad w.r.t. control variables
        control_variables = free[n_states*n_timesteps:]
        grad_control = eta * control_cost_weight * control_variables * np.exp(-beta*timesteps)
        grad[n_states*n_timesteps:] = grad_control

        return grad

    # Specify the symbolic instance constraints (initial and end conditions)
    if not list(w_stud_0):
        w_stud_0 = np.zeros_like(w_teach)
    instance_constraints = []
    instance_constraints = tuple([w_stud_sym[i](0.0) - w_stud_0[i] for i in range(dim_input)])

    # Create an optimization problem
    prob = Problem(obj, obj_grad, eom, state_symbols, n_timesteps, eta,
                   instance_constraints=instance_constraints,
                   known_parameter_map=par_map,
                   known_trajectory_map=trajectory_map,
                   bounds={a(t): (a_min, a_max)})

    # Use a random positive initial guess
    initial_guess = np.random.randn(prob.num_free)

    # Find the optimal solution
    solution, info = prob.solve(initial_guess)
    w_stud_dynamics = solution[:n_timesteps*n_states].reshape((n_states, n_timesteps)).T
    a_dynamics = solution[n_states*n_timesteps:]

    # Compute costs and distance dynamics
    nef_cost_dynamics = np.zeros(n_timesteps)
    per_cost_dynamics = np.zeros(n_timesteps)
    cum_cost_dynamics = np.zeros(n_timesteps)
    d_dynamics = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        w = w_stud_dynamics[t]
        x = x_incoming[t]
        input_s = np.dot(w, x.T)/np.sqrt(dim_input)
        labels_stud = perceptron(input_s, activation=activation)
        input_o = np.dot(w_target, x.T)/np.sqrt(dim_input)
        labels_target = perceptron(input_o, activation=activation)
        nef_cost_dynamics[t] = 0.5 * (labels_stud-labels_target)**2
        per_cost_dynamics[t] = 0.5 * control_cost_weight * a_dynamics[t]**2
        if t==0:
            cum_cost_dynamics[t] = nef_cost_dynamics[t] + per_cost_dynamics[t]
        else:
            discount = np.exp(-beta*timesteps[t])
            cum_cost_dynamics[t] = cum_cost_dynamics[t-1] + (nef_cost_dynamics[t] + per_cost_dynamics[t])*discount
        # relative distance
        input_s_test = np.dot(w, x_test.T)/(dim_input**0.5)
        label_s_test = perceptron(input_s_test, activation=activation)
        d_student_teach = 0.5*np.mean((label_s_test-label_t_test)**2)
        d_dynamics[t] = (d_student_teach/d_target_teach)**0.5

    # Results dictionary
    results = {}
    results['w_dynamics'] = w_stud_dynamics
    results['a_dynamics'] = a_dynamics
    results['nef_cost_dynamics'] = nef_cost_dynamics
    results['per_cost_dynamics'] = per_cost_dynamics
    results['cum_cost_dynamics'] = cum_cost_dynamics
    results['d_dynamics'] = d_dynamics

    return results


# ************* 2-layer NN (NN2L) ***************** #

def exp_clairv_NN2L(x_incoming, x_test, W_stud_0, v_stud_0, W_teach, v_teach, W_target, v_target,
                    dim_input, eta, control_cost_weight=1., a_min=0, a_max=1, beta=0.001, activation='Erf',
                    train_first_layer=True, train_second_layer=True, initial_guess=[], output_scaling='inv_sqroot'):

    # N state variables
    n_weights_1L = W_teach.size
    n_weights_2L = v_teach.size
    n_states = n_weights_1L
    if train_second_layer:
        n_states += n_weights_2L

    # Time grid
    n_timesteps = x_incoming.shape[0]
    duration = eta * (n_timesteps-1)
    timesteps = np.linspace(0, duration, n_timesteps) # t = 0, 1*eta, ...n_timesteps*eta

    # Control cost weight
    control_cost_pref = control_cost_weight
    label_t_test = NN2L(x_test, W_teach, v_teach, dim_input, n_weights_2L,
                        activation=activation, output_scaling=output_scaling)
    label_o_test = NN2L(x_test, W_target, v_target, dim_input, n_weights_2L,
                        activation=activation, output_scaling=output_scaling)
    error_target_teach = np.mean((label_o_test-label_t_test)**2)
    d_target_teach = 0.5 * error_target_teach
    control_cost_weight = control_cost_pref * error_target_teach

    # Symbols
    t = sym.symbols('t')
    x = sym.symbols('x0:%d' % (dim_input), cls=sym.Function)
    W_stud_sym = sym.symbols('W_s0:%d' % (n_weights_1L), cls=sym.Function)
    if train_second_layer:
        v_stud_sym = sym.symbols('v_s0:%d' % (n_weights_2L), cls=sym.Function)
    a = sym.symbols('a', cls=sym.Function)

    # Symbols grouped
    state_symbols_1 = [W_stud_sym[i](t) for i in range(n_weights_1L)]
    state_symbols_2 = []
    if train_second_layer:
        state_symbols_2 = [v_stud_sym[i](t) for i in range(n_weights_2L)]
    state_symbols = tuple(state_symbols_1 + state_symbols_2)
    specified_symbols = (a(t), ) # useful?

    # Trajectory map for x_incoming
    trajectory_map = {}
    for dim in range(dim_input):
        trajectory_map[x[dim](t)] = x_incoming[:, dim]

    # Input student
    activation_L1_stud_sym = []
    for j in range(n_weights_2L):
        input_s = 0
        for i in range(dim_input):
            input_s += W_stud_sym[j*dim_input+i](t)*x[i](t)/dim_input**0.5
        activation_L1_stud_sym.append(perceptron_sym(input_s, activation=activation))

    # Input teacher
    activation_L1_teach_sym = []
    for j in range(n_weights_2L):
        input_t = 0
        for i in range(dim_input):
            input_t += W_teach[j][i]*x[i](t)/dim_input**0.5
        activation_L1_teach_sym.append(perceptron_sym(input_t, activation=activation))

    # Input target
    activation_L1_target_sym = []
    for j in range(n_weights_2L):
        input_o = 0
        for i in range(dim_input):
            input_o += W_target[j][i]*x[i](t)/dim_input**0.5
        activation_L1_target_sym.append(perceptron_sym(input_o, activation=activation))

    # Pre-factor
    if output_scaling=='inv':
        pref_hlayer_size = hlayer_size
    elif output_scaling=='inv_sqroot':
        pref_hlayer_size = np.sqrt(hlayer_size)
    else:
        raise ValueError('output_scaling not understood')

    # Output (labels)
    label_s = 0
    if train_second_layer:
        for j in range(n_weights_2L):
            label_s += v_stud_sym[j](t)*activation_L1_stud_sym[j]/pref_hlayer_size
    else:
        for j in range(n_weights_2L):
            label_s += v_stud_0[j]*activation_L1_stud_sym[j]/pref_hlayer_size
    label_t = 0
    for j in range(n_weights_2L):
        label_t += v_teach[j]*activation_L1_teach_sym[j]/pref_hlayer_size
    label_o = 0
    for j in range(n_weights_2L):
        label_o += v_target[j]*activation_L1_target_sym[j]/pref_hlayer_size
    label_dagger = label_t*(1-a(t)) + label_o*a(t)

    # Loss function
    Loss_fct = (label_s - label_dagger)**2 / 2.

    # Loss function grad
    Loss_fct_diff_W = tuple([sym.diff(Loss_fct, W_stud_sym[i](t)) for i in range(n_weights_1L)])
    if train_second_layer:
        Loss_fct_diff_v = tuple([sym.diff(Loss_fct, v_stud_sym[i](t)) for i in range(n_weights_2L)])

    # Equations of motion
    eom_list_W = [W_stud_sym[i](t).diff() + Loss_fct_diff_W[i] for i in range(n_weights_1L)]
    eom_list_v = []
    if train_second_layer:
        eom_list_v = [v_stud_sym[i](t).diff() + Loss_fct_diff_v[i] for i in range(n_weights_2L)]
    eom_list = eom_list_W + eom_list_v
    eom = sym.Matrix(eom_list)

    # Specify the known system parameters
    par_map = OrderedDict()

    # Specify the objective function and it's gradient
    def obj(free):
        '''
           Objective function to be minimized: discounted sum of action and nefarious costs.
           Takes as input the free vector, which stores student parameters and actions at all times:

           -> free = [state_0_t0, ...state_0_tend, state_1_t0, ...state_1_tend, ...state_last_tend,
                      control_t0, ...control_tend]
                      k = j*dim_input+i < n_weights_1L
                      state_k_tj = W_stud_ji_tj
                      k = n_weights_1L+i < n_weights_1L + n_weights_2L
                      state_k_tj = v_stud_i_tj
                      control_tj = a_tj : action is "control variable"
              so the first n_states * n_timesteps elements in free are state variables at all times,
              followed by n_timesteps elements that are the control variable at all times.

              nef_cost_tj = 0.5 * np.dot(state_tj-w_target, x_incoming[j])**2 / dim_input
              per_cost_tj = 0.5 * control_cost_weight * control_tj**2
              total_cost = Sum_j (per_cost_tj+nef_cost_tj) * eta * exp(-beta * eta * j)
        '''

        Cost = Cost_control = Cost_nef = 0

        # Cost due to state variables
        state_variables = free[:n_states*n_timesteps]
        W_states = state_variables[:n_weights_1L*n_timesteps]
        W_states = W_states.reshape(n_weights_1L, n_timesteps).T
        W_states = W_states.reshape(n_timesteps, n_weights_2L, dim_input)
        if train_second_layer:
            v_states = state_variables[n_weights_1L*n_timesteps:]
            v_states = v_states.reshape(n_weights_2L, n_timesteps).T
        else:
            v_states = np.expand_dims(v_stud_0, axis=0).repeat(n_timesteps, axis=0)
        labels_s_l = NN2L(x_incoming, W_states, v_states, dim_input,
                          n_weights_2L, activation=activation, output_scaling=output_scaling)
        labels_o_l = NN2L(x_incoming, W_target, v_target, dim_input,
                          n_weights_2L, activation=activation, output_scaling=output_scaling)
        Cost_nef = 0.5*(labels_s_l-labels_o_l)**2
        Cost_nef = np.sum(eta * Cost_nef * np.exp(-beta*timesteps))

        # Cost due to control variables
        control_variables = free[n_states*n_timesteps:]
        Cost_control = 0.5 * control_cost_weight * control_variables**2
        Cost_control = np.sum(eta * Cost_control * np.exp(-beta*timesteps))

        return Cost_nef + Cost_control

    def obj_grad(free):
        '''
           Gradient of the objective function with respect to the free vector.
        '''

        grad = np.zeros_like(free)

        # Grad w.r.t. state variables W
        state_variables = free[:n_states*n_timesteps]
        W_states = state_variables[:n_weights_1L*n_timesteps]
        W_states = W_states.reshape(n_weights_1L, n_timesteps).T
        W_states = W_states.reshape(n_timesteps, n_weights_2L, dim_input)
        if train_second_layer:
            v_states = state_variables[n_weights_1L*n_timesteps:]
            v_states = v_states.reshape(n_weights_2L, n_timesteps).T
        else:
            v_states = np.expand_dims(v_stud_0, axis=0).repeat(n_timesteps, axis=0)
        labels_s_l = NN2L(x_incoming, W_states, v_states, dim_input,
                          n_weights_2L, activation=activation, output_scaling=output_scaling)
        labels_o_l = NN2L(x_incoming, W_target, v_target, dim_input,
                          n_weights_2L, activation=activation, output_scaling=output_scaling)
        # Const. factor
        const_factor = eta/(pref_hlayer_size*np.sqrt(dim_input))
        # Time factors
        time_factors = (labels_s_l-labels_o_l) * np.exp(-beta*timesteps)
        time_factors_exp = np.expand_dims(time_factors, axis=[-2, -1])
        # Second layer and time factors
        input_1L = np.squeeze(np.matmul(W_states, np.expand_dims(x_incoming, axis=-1))/np.sqrt(dim_input))
        secondL_time_factors = d_perceptron(input_1L, activation=activation).reshape(*v_states.shape) * v_states
        secondL_time_factors_exp = np.expand_dims(secondL_time_factors, axis=[-1])
        # First layer and time factors
        firstL_time_factors = x_incoming
        firstL_time_factors_exp = np.expand_dims(firstL_time_factors, axis=[-2])
        # grad_W_factors
        grad_W = const_factor * time_factors_exp * firstL_time_factors_exp * secondL_time_factors_exp
        grad_W = grad_W.reshape(n_timesteps, n_weights_1L).T
        grad_W = grad_W.reshape(-1,)
        grad[:n_timesteps*n_weights_1L] = grad_W

        # Grad w.r.t. state variables v
        if train_second_layer:
            # Const. factor
            const_factor = eta/pref_hlayer_size
            # Time factors
            time_factors = (labels_s_l-labels_o_l) * np.exp(-beta*timesteps)
            time_factors_exp = np.expand_dims(time_factors, axis=-1)
            # Second layer and time factors
            input_1L = np.squeeze(np.matmul(W_states, np.expand_dims(x_incoming, axis=-1))/np.sqrt(dim_input), axis=-1)
            secondL_time_factors = perceptron(input_1L, activation=activation)
            # grad_v_factors
            grad_v = const_factor * time_factors_exp * secondL_time_factors
            grad_v = grad_v.reshape(n_timesteps, n_weights_2L).T
            grad_v = grad_v.reshape(-1,)
            grad[n_timesteps*n_weights_1L:n_timesteps*(n_weights_1L+n_weights_2L)] = grad_v

        # Grad w.r.t. control variable a
        control_variables = free[n_states*n_timesteps:]
        grad_a =  eta * control_cost_weight * control_variables * np.exp(-beta*timesteps)
        grad[n_timesteps*n_states:] = grad_a

        return grad

    # Specify the symbolic instance constraints (initial and end conditions)
    instance_constraints_W = [W_stud_sym[j*dim_input+i](0.0) - W_stud_0[j][i] for j in range(n_weights_2L) for i in range(dim_input)]
    instance_constraints_v = []
    if train_second_layer:
        instance_constraints_v = [v_stud_sym[i](0.0) - v_stud_0[i] for i in range(n_weights_2L)]
    instance_constraints = tuple(instance_constraints_W+instance_constraints_v)

    # Create an optimization problem
    prob = Problem(obj, obj_grad, eom, state_symbols, n_timesteps, eta,
                      instance_constraints=instance_constraints,
                      known_parameter_map=par_map,
                      known_trajectory_map=trajectory_map,
                      bounds={a(t): (a_min, a_max)})

    # Use a random positive initial guess
    if len(initial_guess)==0:
        initial_guess = np.random.randn(prob.num_free)

    # Find the optimal solution
    solution, info = prob.solve(initial_guess)
    state_variables = solution[:n_states*n_timesteps]
    W_states_dynamics = state_variables[:n_weights_1L*n_timesteps]
    W_states_dynamics = W_states_dynamics.reshape(n_weights_1L, n_timesteps).T
    W_states_dynamics = W_states_dynamics.reshape(n_timesteps, n_weights_2L, dim_input)
    if train_second_layer:
        v_states_dynamics = state_variables[n_weights_1L*n_timesteps:]
        v_states_dynamics = v_states_dynamics.reshape(n_weights_2L, n_timesteps).T
    else:
        v_states_dynamics = np.expand_dims(v_stud_0, axis=0).repeat(n_timesteps, axis=0)
    a_dynamics = solution[n_states*n_timesteps:]

    # Compute costs and distance dynamics
    nef_cost_dynamics = np.zeros(n_timesteps)
    per_cost_dynamics = np.zeros(n_timesteps)
    cum_cost_dynamics = np.zeros(n_timesteps)
    d_dynamics = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        W_stud_t = W_states_dynamics[t]
        v_stud_t = v_states_dynamics[t]
        x = x_incoming[t]
        labels_stud = NN2L(x, W_stud_t, v_stud_t, dim_input, n_weights_2L,
                           activation=activation, output_scaling=output_scaling)
        labels_target = NN2L(x, W_target, v_target, dim_input, n_weights_2L,
                             activation=activation, output_scaling=output_scaling)
        nef_cost_dynamics[t] = 0.5 * (labels_stud-labels_target)**2
        per_cost_dynamics[t] = 0.5 * control_cost_weight * a_dynamics[t]**2
        if t==0:
            cum_cost_dynamics[t] = nef_cost_dynamics[t] + per_cost_dynamics[t]
        else:
            discount = np.exp(-beta*timesteps[t])
            cum_cost_dynamics[t] = cum_cost_dynamics[t-1] + (nef_cost_dynamics[t] + per_cost_dynamics[t])*discount
        # relative distance
        label_s_test = NN2L(x_test, W_stud_t, v_stud_t, dim_input, n_weights_2L,
                            activation=activation, output_scaling=output_scaling)
        d_student_teach = 0.5 * np.mean((label_s_test-label_t_test)**2)
        d_dynamics[t] = (d_student_teach/d_target_teach)**0.5

    # Results dictionary
    results = {}
    #results['W_dynamics'] = W_states_dynamics
    results['v_dynamics'] = v_states_dynamics
    results['a_dynamics'] = a_dynamics
    results['nef_cost_dynamics'] = nef_cost_dynamics
    results['per_cost_dynamics'] = per_cost_dynamics
    results['cum_cost_dynamics'] = cum_cost_dynamics
    results['d_dynamics'] = d_dynamics

    return results
