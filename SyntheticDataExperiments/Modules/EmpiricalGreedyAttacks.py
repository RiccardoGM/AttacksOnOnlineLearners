## Import statements ##

import numpy as np
import scipy as sp



# ************************************************ #
#                                                  #
#               Auxiliary functions                #
#                                                  #
# ************************************************ #

# ***************** Soft ReLU ********************* #

''' softReLU and derivative (step function) '''

s_SReLU = .2

def SReLU(x, s=s_SReLU):
    return x * 0.5 * sp.special.erfc(-(x/s)/np.sqrt(2)) + s * np.exp(-(x/s)**2/2.)/np.sqrt(2*np.pi)

def dSReLU(x, s=s_SReLU):
    return 0.5 * sp.special.erfc(-(x/s)/np.sqrt(2))


# ******************* ReLU *********************** #

''' ReLU and derivative (step function) '''

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


# ************** Error function ****************** #

''' Erf and derivative '''

def Erf(x):
    return sp.special.erf(x)

def dErf(x):
    return (2./np.sqrt(np.pi)) * np.exp(-x**2)



# ************************************************* #
#                                                   #
#                      Models                       #
#                                                   #
# ************************************************* #

# ***************** Perceptron ******************** #

''' Linear layer '''

def linear(x, w):

    '''this function assumes w with shape (n_models, dim_input)'''

    w = np.atleast_2d(w)
    if len(w.shape)<2:
        w = np.expand_dims(w, axis=0)
    dim_input = w.shape[1]
    output = np.dot(x, w.T)/dim_input**0.5

    return output


'''  Perceptron '''

def perceptron(x, w=[], activation='Erf'):

    if len(w)>0:
        x = linear(x, w)

    if activation=='Erf':
        output = Erf(x/(2**0.5))
    elif activation=='ReLU':
        output = ReLU(x)
    elif activation=='SReLU':
        output = SReLU(x)
    elif activation=='Linear':
        output = x
    else:
        raise ValueError('Activation must be one of the following: Erf, ReLU, SReLU, Linear')

    return output


''' Derivative perceptron '''

def d_perceptron(x, w=[], dim_input=None, activation='Erf'):

    if len(w)>0:
        x = linear(x, w)

    if activation=='Erf':
        output = (1/np.sqrt(2)) * dErf(x/np.sqrt(2))
    elif activation=='ReLU':
        output = dReLU(x)
    elif activation=='SReLU':
        output = dSReLU(x)
    elif activation=='Linear':
        output = np.ones_like(x)
    else:
        raise ValueError('Activation must be one of the following: Erf, ReLU, SReLU, Linear')

    return output


# ************* 2-layer NN (NN2L) ***************** #

''' 2-layer NN '''

def NN2L(X, W, v, activation='Erf', output_scaling='inv_sqroot'):

    if len(X.shape)<2:
        raise ValueError('NN2L admits X inputs as 2D arrays of shape (n_samples, dim_input)')

    hlayer_size = v.shape[-1]
    dim_input = W.shape[-1]

    if output_scaling=='inv':
        pref_hlayer_size = hlayer_size #np.sqrt(hlayer_size)
    elif output_scaling=='inv_sqroot':
        pref_hlayer_size = np.sqrt(hlayer_size)
    else:
        raise ValueError('output_scaling not understood')
    in_L1 = np.matmul(W, X.T)/np.sqrt(dim_input)
    in_L2 = perceptron(in_L1, activation=activation)
    output = np.sum(in_L2 * np.expand_dims(v, axis=-1), axis=-2)/pref_hlayer_size #np.sqrt(hlayer_size)
    #output = np.squeeze(output)

    return output



# ************************************************* #
#                                                   #
#                 Policy functions                  #
#                                                   #
# ************************************************* #

# ***************** Perceptron ******************** #

def a_greedy_perceptron(w_stud, w_target, w_teach, x_batch, x_buffer, dim_input, eta,
                        weight_future=[], a_min=0, a_max=1, n_gridpoints=int(1e3),
                        control_cost_weight=1., activation='Erf'):

    '''
        If len(weight_future)>1, then it has to match n_students.

        w_stud: 2d array of shape (n_students, dim_input)
        w_teach, w_target: 1d arrays of len dim_input
        x_batch: 2d array of shape (n_batches, dim_input)
        weight_future: 1d array of len 1 or n_students

    '''

    if len(np.atleast_1d(weight_future))==0:
        weight_future = dim_input/eta

    # weight_future at least 1D
    weight_future = np.atleast_1d(weight_future)

    # w_stud at least 2D
    if len(w_stud.shape)==1:
        w_stud = w_stud.reshape(1, -1)
    n_students = w_stud.shape[0]

    # Grid of possible actions
    a_grid = np.linspace(a_min, a_max, n_gridpoints)

    # Control cost
    control_cost = 0.5*control_cost_weight*a_grid**2

    # Next student vector
    w_stud_next_grid = student_update_perceptron(w_stud,
                                                   w_teach,
                                                   w_target,
                                                   x_batch,
                                                   a_grid,
                                                   eta,
                                                   dim_input,
                                                   activation)

    # Nefarious cost
    input_s = np.matmul(w_stud_next_grid, x_buffer.T)/(dim_input**0.5)
    input_o = np.dot(w_target, x_buffer.T)/(dim_input**0.5)
    if activation=='Erf':
        label_s = Erf(input_s/(2**0.5))
        label_o = Erf(input_o/(2**0.5))
    elif activation=='ReLU':
        label_s = ReLU(input_s)
        label_o = ReLU(input_o)
    elif activation=='SReLU':
        label_s = SReLU(input_s)
        label_o = SReLU(input_o)
    elif activation=='Linear':
        label_s = input_s
        label_o = input_o
    shape_label_s = np.ones(len(label_s.shape)-1, dtype=int)
    nef_cost = np.mean((label_s-label_o.reshape(*shape_label_s,-1))**2, axis=-1)/2.

    # Total cost
    cost = control_cost.reshape(-1,1) + weight_future.reshape(1,-1)*nef_cost.reshape(n_gridpoints, n_students)

    # Find a_greedy
    a_idx = np.argmin(cost, axis=0)
    a_greedy = a_grid[a_idx]

    return a_greedy


# ************* 2-layer NN (NN2L) ***************** #

def a_greedy_NN2L(W_stud, v_stud, W_target, v_target, W_teach, v_teach, x_batch, x_buffer, dim_input, eta,
                  weight_future=[], a_min=0, a_max=1, n_gridpoints=int(1e3), control_cost_weight=1.,
                  activation='Erf', train_first_layer=True, train_second_layer=True, output_scaling='inv_sqroot'):

    if len(np.atleast_1d(weight_future))==0:
        weight_future = dim_input/eta

    # weight_future at least 1D
    weight_future = np.atleast_1d(weight_future)

    # Size hidden layer
    hlayer_size = len(v_stud)

    # w_stud at least 3D
    if len(W_stud.shape)==2:
        W_stud = np.expand_dims(W_stud, axis=0)
    n_students = W_stud.shape[0]

    # Grid of possible actions
    a_grid = np.linspace(a_min, a_max, n_gridpoints)

    # Control cost
    control_cost = 0.5*control_cost_weight*a_grid**2

    # Next student vector
    W_stud_next_grid, v_stud_next_grid = student_update_NN2L(W_stud,
                                                             v_stud,
                                                             W_target,
                                                             v_target,
                                                             W_teach,
                                                             v_teach,
                                                             x_batch,
                                                             a_grid,
                                                             eta,
                                                             dim_input,
                                                             activation,
                                                             train_first_layer=train_first_layer,
                                                             train_second_layer=train_second_layer,
                                                             output_scaling=output_scaling)

    # Nefarious cost
    label_s = NN2L(x_buffer, W_stud_next_grid, v_stud_next_grid, activation, output_scaling=output_scaling)
    label_o = NN2L(x_buffer, W_target, v_target, activation, output_scaling=output_scaling)
    shape_label_s = np.ones(len(label_s.shape)-1, dtype=int)
    nef_cost = np.mean((label_s-label_o.reshape(*shape_label_s,-1))**2, axis=-1)/2.

    # Total cost
    cost = control_cost.reshape(-1,1) + weight_future.reshape(1,-1)*nef_cost.reshape(n_gridpoints, n_students)

    # Find a_greedy
    a_idx = np.argmin(cost, axis=0)
    a_greedy = a_grid[a_idx]

    return a_greedy



# ************************************************* #
#                                                   #
#                    Update rule                    #
#                                                   #
# ************************************************* #

# ***************** Perceptron ******************** #

def student_update_perceptron(w_stud, w_teach, w_target, x_batch, a, eta, dim_input, activation='Erf', actions_students_match=False, fraction_poisoned=False, fraction_poisoned_samples=1.):
    '''
        a: 1d array, each value to be used for each w_stud vector
        w_stud: 2d array of shape (n_students, dim_input)
        w_teach, w_target: 1d arrays of len dim_input
        x_batch: 2d array of shape (n_batches, dim_input)
        actions_students_match: if True, len(a) and n_students should match,
                                and w_stud[i] is updated with a[i].
    '''

    # Check/fix shapes
    a = np.atleast_1d(a)
    if len(w_stud.shape)<2:
        w_stud = np.expand_dims(w_stud, axis=0)
    if len(x_batch.shape)<2:
        x_batch = np.expand_dims(x_batch, axis=0)

    # N. students
    n_students = w_stud.shape[0]

    # N values in a
    n_a_values = len(a)

    # Batch size
    batch_size = x_batch.shape[0]

    # Next student vector
    input_s = np.dot(w_stud, x_batch.T)/(dim_input**0.5)
    input_t = np.dot(w_teach, x_batch.T)/(dim_input**0.5)
    input_o = np.dot(w_target, x_batch.T)/(dim_input**0.5)
    # Erf
    if activation=='Erf':
        label_s = Erf(input_s/(2**0.5))
        label_t = Erf(input_t/(2**0.5))
        label_o = Erf(input_o/(2**0.5))
        x_batch_exp = np.repeat(np.expand_dims(x_batch, axis=0), n_students, axis=0)
        v1_0_factors = np.sqrt(1/(2*dim_input)) * (label_s-label_t.reshape(1,-1)) * dErf(input_s/(2**0.5))
        v1_0_factors_exp = np.repeat(np.expand_dims(v1_0_factors, axis=-1), dim_input, axis=-1)
        v1_0 = np.mean(v1_0_factors_exp * x_batch_exp, axis=1)
        if fraction_poisoned:
            n_poisoned_samples = int(fraction_poisoned_samples * batch_size)
            label_t = label_t[-n_poisoned_samples:]
            label_o = label_o[-n_poisoned_samples:]
            x_batch_exp = np.repeat(np.expand_dims(x_batch[-n_poisoned_samples:], axis=0), n_students, axis=0)
        v2_0_factors = np.sqrt(1/(2*dim_input)) * (label_t-label_o) * dErf(input_s/(2**0.5))
        v2_0_factors_exp = np.repeat(np.expand_dims(v2_0_factors, axis=-1), dim_input, axis=-1)
        v2_0 = np.mean(v2_0_factors_exp * x_batch_exp, axis=1)
        if fraction_poisoned:
            v2_0 = fraction_poisoned_samples * v2_0
    # ReLU
    elif activation=='ReLU':
        label_s = ReLU(input_s)
        label_t = ReLU(input_t)
        label_o = ReLU(input_o)
        x_batch_exp = np.repeat(np.expand_dims(x_batch, axis=0), n_students, axis=0)
        v1_0_factors = (1/np.sqrt(dim_input)) * (label_s-label_t.reshape(1,-1)) * dReLU(input_s)
        v1_0_factors_exp = np.repeat(np.expand_dims(v1_0_factors, axis=-1), dim_input, axis=-1)
        v1_0 = np.mean(v1_0_factors_exp * x_batch_exp, axis=1)
        if fraction_poisoned:
            n_poisoned_samples = int(fraction_poisoned_samples * batch_size)
            label_t = label_t[-n_poisoned_samples:]
            label_o = label_o[-n_poisoned_samples:]
            x_batch_exp = np.repeat(np.expand_dims(x_batch[-n_poisoned_samples:], axis=0), n_students, axis=0)
        v2_0_factors = (1/np.sqrt(dim_input)) * (label_t-label_o) * dReLU(input_s)
        v2_0_factors_exp = np.repeat(np.expand_dims(v2_0_factors, axis=-1), dim_input, axis=-1)
        v2_0 = np.mean(v2_0_factors_exp * x_batch_exp, axis=1)
        if fraction_poisoned:
            v2_0 = fraction_poisoned_samples * v2_0
    # SReLU
    elif activation=='SReLU':
        label_s = SReLU(input_s)
        label_t = SReLU(input_t)
        label_o = SReLU(input_o)
        x_batch_exp = np.repeat(np.expand_dims(x_batch, axis=0), n_students, axis=0)
        v1_0_factors = (1/np.sqrt(dim_input)) * (label_s-label_t.reshape(1,-1)) * dSReLU(input_s)
        v1_0_factors_exp = np.repeat(np.expand_dims(v1_0_factors, axis=-1), dim_input, axis=-1)
        v1_0 = np.mean(v1_0_factors_exp * x_batch_exp, axis=1)
        if fraction_poisoned:
            n_poisoned_samples = int(fraction_poisoned_samples * batch_size)
            label_t = label_t[-n_poisoned_samples:]
            label_o = label_o[-n_poisoned_samples:]
            x_batch_exp = np.repeat(np.expand_dims(x_batch[-n_poisoned_samples:], axis=0), n_students, axis=0)
        v2_0_factors = (1/np.sqrt(dim_input)) * (label_t-label_o) * dSReLU(input_s)
        v2_0_factors_exp = np.repeat(np.expand_dims(v2_0_factors, axis=-1), dim_input, axis=-1)
        v2_0 = np.mean(v2_0_factors_exp * x_batch_exp, axis=1)
        if fraction_poisoned:
            v2_0 = fraction_poisoned_samples * v2_0
    # Linear
    elif activation=='Linear':
        label_s = input_s
        label_t = input_t
        label_o = input_o
        x_batch_exp = np.repeat(np.expand_dims(x_batch, axis=0), n_students, axis=0)
        v1_0_factors = (1/np.sqrt(dim_input)) * (label_s-label_t.reshape(1,-1))
        v1_0_factors_exp = np.repeat(np.expand_dims(v1_0_factors, axis=-1), dim_input, axis=-1)
        v1_0 = np.mean(v1_0_factors_exp * x_batch_exp, axis=1)
        if fraction_poisoned:
            n_poisoned_samples = int(fraction_poisoned_samples * batch_size)
            label_t = label_t[-n_poisoned_samples:]
            label_o = label_o[-n_poisoned_samples:]
            x_batch_exp = np.repeat(np.expand_dims(x_batch[-n_poisoned_samples:], axis=0), n_students, axis=0)
        v2_0_factors = (1/np.sqrt(dim_input)) * (label_t-label_o)
        v2_0_factors_exp = np.repeat(np.expand_dims(v2_0_factors, axis=-1), dim_input, axis=-1)
        v2_0 = np.mean(v2_0_factors_exp * x_batch_exp, axis=1)
        if fraction_poisoned:
            v2_0 = fraction_poisoned_samples * v2_0
    else:
        raise ValueError('Activation must be one of the following: Erf, ReLU, SReLU, Linear')
    # v1 and v2
    v1 = w_stud - eta*v1_0
    v2 = - eta*v2_0
    if actions_students_match:
        w_stud_next = v1 + a.reshape(-1,1)*v2
        if n_students==1:
            w_stud_next = w_stud_next[0]
    else:
        v1_exp = np.expand_dims(v1, axis=0).repeat(n_a_values, axis=0)
        v2_exp = np.expand_dims(v2, axis=0).repeat(n_a_values, axis=0)
        a_exp = np.expand_dims(a, axis=[-2, -1]).repeat(n_students, axis=-2).repeat(dim_input, axis=-1)
        w_stud_next = v1_exp + a_exp*v2_exp
        if n_a_values==1:
            w_stud_next = w_stud_next[0]
            if n_students==1:
                w_stud_next = w_stud_next[0]
        else:
            if n_students==1:
                w_stud_next = w_stud_next[:, 0, :]

    return w_stud_next


# ************* 2-layer NN (NN2L) ***************** #

def student_update_NN2L(W_stud, v_stud, W_target, v_target, W_teach, v_teach, x_batch, a, eta, dim_input, activation='Erf', actions_students_match=False, train_first_layer=True, train_second_layer=True, output_scaling='inv_sqroot'):
    '''
        a: 1d array
        W_teach, W_target: 2d arrays (hidden weights)
        W_stud: 3d array of shape (n_students, dim_input, hlayer_size)
        v_teach, v_target: 1d arrays (output weights)
        v_stud: 2d array of shape (n_students, hlayer_size)
        x_batch: 2d array of shape (n_batches, dim_input)
    '''

    # Check/fix shapes
    a = np.atleast_1d(a)
    if len(W_stud.shape)<2:
        W_stud = np.expand_dims(W_stud, axis=0)
    if len(W_stud.shape)<3:
        W_stud = np.expand_dims(W_stud, axis=0)
    if len(W_teach.shape)<2:
        W_teach = np.expand_dims(W_teach, axis=0)
    if len(W_teach.shape)<3:
        W_teach = np.expand_dims(W_teach, axis=0)
    if len(W_target.shape)<2:
        W_target = np.expand_dims(W_target, axis=0)
    if len(W_target.shape)<3:
        W_target = np.expand_dims(W_target, axis=0)
    v_stud = np.atleast_1d(v_stud)
    if len(v_stud.shape)<2:
        v_stud = np.expand_dims(v_stud, axis=0)
    v_teach = np.atleast_1d(v_teach)
    if len(v_teach.shape)<2:
        v_teach = np.expand_dims(v_teach, axis=0)
    v_target = np.atleast_1d(v_target)
    if len(v_target.shape)<2:
        v_target = np.expand_dims(v_target, axis=0)
    x_batch = np.atleast_2d(x_batch)

    # Save copy of student weights
    W_stud_input = W_stud.copy()
    v_stud_input = v_stud.copy()

    # N. students
    n_students = W_stud.shape[0]

    # Hidden layer size and scaling
    hlayer_size = W_stud.shape[1]
    if output_scaling=='inv':
        pref_hlayer_size = hlayer_size #np.sqrt(hlayer_size)
    elif output_scaling=='inv_sqroot':
        pref_hlayer_size = np.sqrt(hlayer_size)
    else:
        raise ValueError('output_scaling not understood')

    # N values in a
    n_a_values = len(a)

    # Batch size
    batch_size = x_batch.shape[0]

    # labels student
    input_L1_s = np.dot(W_stud, x_batch.T)/np.sqrt(dim_input)
    input_L2_s = perceptron(input_L1_s, activation=activation)
    v_stud_exp = np.repeat(np.expand_dims(v_stud, axis=-1), batch_size, axis=-1)
    label_s = np.sum(v_stud_exp*input_L2_s, axis=1)/pref_hlayer_size

    # labels teacher
    input_L1_t = np.dot(W_teach, x_batch.T)/np.sqrt(dim_input)
    input_L2_t = perceptron(input_L1_t, activation=activation)
    v_teach_exp = np.repeat(np.expand_dims(v_teach, axis=-1), batch_size, axis=-1)
    label_t = np.sum(v_teach_exp*input_L2_t, axis=1)/pref_hlayer_size

    # labels target
    input_L1_o = np.dot(W_target, x_batch.T)/np.sqrt(dim_input)
    input_L2_o = perceptron(input_L1_o, activation=activation)
    v_target_exp = np.repeat(np.expand_dims(v_target, axis=-1), batch_size, axis=-1)
    label_o = np.sum(v_target_exp*input_L2_o, axis=1)/pref_hlayer_size

    # Expand/repeat batch
    x_batch_exp = np.repeat(np.expand_dims(x_batch, axis=0), n_students, axis=0)
    x_batch_exp = np.repeat(np.expand_dims(x_batch_exp, axis=1), hlayer_size, axis=1)

    # Vectors v1 and v2 for W_stud
    v1_0_factors = np.repeat(np.expand_dims((label_s-label_t), axis=1), hlayer_size, axis=1)
    v1_0_factors = (1/(pref_hlayer_size*dim_input**0.5)) * v1_0_factors * v_stud_exp * d_perceptron(input_L1_s, activation=activation)
    v1_0_factors_exp = np.repeat(np.expand_dims(v1_0_factors, axis=-1), dim_input, axis=-1)
    v1_0 = np.mean(v1_0_factors_exp*x_batch_exp, axis=-2)
    v2_0_factors = np.repeat(np.expand_dims((label_t-label_o), axis=1), hlayer_size, axis=1)
    v2_0_factors = (1/(pref_hlayer_size*dim_input**0.5)) * v2_0_factors * v_stud_exp * d_perceptron(input_L1_s, activation=activation)
    v2_0_factors_exp = np.repeat(np.expand_dims(v2_0_factors, axis=-1), dim_input, axis=-1)
    v2_0 = np.mean(v2_0_factors_exp*x_batch_exp, axis=-2)
    v1 = W_stud - eta*v1_0
    v2 = - eta*v2_0

    # Vectors v3 and v4 for v_stud
    v3_0_factors = np.repeat(np.expand_dims((label_s-label_t), axis=1), hlayer_size, axis=1)
    v3_0_factors = (1/pref_hlayer_size) * v3_0_factors * input_L2_s  #/np.sqrt(hlayer_size)
    v3_0 = np.mean(v3_0_factors, axis=-1)
    v4_0_factors = np.repeat(np.expand_dims((label_t-label_o), axis=1), hlayer_size, axis=1)
    v4_0_factors = (1/pref_hlayer_size) * v4_0_factors * input_L2_s  #/np.sqrt(hlayer_size)
    v4_0 = np.mean(v4_0_factors, axis=-1)
    v3 = v_stud - eta*v3_0
    v4 = - eta*v4_0

    # New parameters
    if actions_students_match:
        ''' W_stud_next: (n_students, hlayer_size, dim_input),
            v_stud_next: (n_students, hlayer_size) '''
        W_stud_next = v1 + a.reshape(-1,1,1)*v2
        v_stud_next = v3 + a.reshape(-1,1)*v4
        if n_students==1:
            ''' W_stud_next: (hlayer_size, dim_input),
                v_stud_next: (hlayer_size) '''
            pass
    else:
        v1_exp = np.expand_dims(v1, axis=0).repeat(n_a_values, axis=0)
        v2_exp = np.expand_dims(v2, axis=0).repeat(n_a_values, axis=0)
        W_stud_next = v1_exp + a.reshape(-1,1,1,1)*v2_exp
        v3_exp = np.expand_dims(v3, axis=0).repeat(n_a_values, axis=0)
        v4_exp = np.expand_dims(v4, axis=0).repeat(n_a_values, axis=0)
        v_stud_next = v3_exp + a.reshape(-1,1,1)*v4_exp
        if n_a_values==1:
            ''' W_stud_next: (n_students, hlayer_size, dim_input),
                v_stud_next: (n_students, hlayer_size) '''
            W_stud_next = W_stud_next[0]
            v_stud_next = v_stud_next[0]
            if n_students==1:
                ''' W_stud_next: (hlayer_size, dim_input),
                    v_stud_next: (hlayer_size) '''
                pass
        else:
            ''' W_stud_next: (n_a_values, n_students, hlayer_size, dim_input),
                v_stud_next: (n_a_values, n_students, hlayer_size) '''
            if n_students==1:
                ''' W_stud_next: (n_a_values, hlayer_size, dim_input),
                    v_stud_next: (n_a_values, hlayer_size) '''
                W_stud_next = W_stud_next[:, 0, :, :]
                v_stud_next = v_stud_next[:, 0, :]

    # Ignore updates if needed
    if (train_first_layer==False):
        W_stud_next = W_stud_input.repeat(W_stud_next.shape[0], axis=0)
    if (train_second_layer==False):
        v_stud_next = v_stud_input.repeat(v_stud_next.shape[0], axis=0)

    return W_stud_next, v_stud_next



# ************************************************* #
#                                                   #
#                    Experiment                     #
#                                                   #
# ************************************************* #

# ***************** Perceptron ******************** #

''' Same perturbation for entire batch '''

def exp_greedy_perceptron(x_incoming, x_past, x_buffer, x_test, eta, w_teach, w_target, dim_input,
                          weight_future=[], w_stud_0=[], a_min=0, a_max=1, beta=0.001, control_cost_weight=1.,
                          batch_size=1, n_gridpoints=int(1e2), buffer_size=250, transient_th=4000,
                          window_steadystate=1000, activation='Erf', opt_pref=False, fut_pref_min=0.1,
                          fut_pref_max=5.1, fut_pref_interval=0.1, n_av=10, fut_pref=1., fraction_poisoned_samples=1.):

    # N. poisoned samples
    n_poisoned_samples = int(fraction_poisoned_samples * batch_size)
    n_clean_samples = batch_size - n_poisoned_samples

    # Initialisation
    if len(np.atleast_1d(weight_future))==0:
        weight_future = dim_input/eta
    if not list(w_stud_0):
        w_stud_0 = np.zeros_like(w_teach)
    w_stud = w_stud_0

    # Control cost weight
    control_cost_pref = control_cost_weight
    input_t_test = np.dot(w_teach, x_test.T)/(dim_input**0.5)
    input_o_test = np.dot(w_target, x_test.T)/(dim_input**0.5)
    label_t_test = perceptron(input_t_test, activation=activation)
    label_o_test = perceptron(input_o_test, activation=activation)
    d_target_teach = np.mean((label_o_test-label_t_test)**2)
    control_cost_weight = control_cost_pref * d_target_teach
    print(control_cost_weight)

    # Teach labels
    t_pred_test = np.sign(label_t_test)
    n_samples_test = x_test.shape[0]

    # Cost vs fut_pref
    fut_pref_grid_opt = np.arange(fut_pref_min, fut_pref_max, fut_pref_interval)
    fut_pref_grid = fut_pref * np.ones_like(fut_pref_grid_opt)
    running_cost_vs_fut_pref = np.zeros(len(np.arange(fut_pref_min, fut_pref_max, fut_pref_interval)))

    #************************************#
    #       Optimize future weight       #
    #************************************#

    if opt_pref:
        n_past_timesteps = int(x_past.shape[0]/batch_size)
        upper_bound_pref_fut = fut_pref_min + (round((fut_pref_max-fut_pref_min)/fut_pref_interval)-1)*fut_pref_interval
        print('Optimising future weight pre-factor between %.4f and %.4f'%(fut_pref_min, upper_bound_pref_fut))
        print('N. grid points for future weight: %d' % len(fut_pref_grid_opt))
        print('N past timesteps:', n_past_timesteps)

        x_past_whole = np.concatenate((x_past, x_buffer), axis=0).copy()
        fut_pref_grid = fut_pref_grid_opt
        running_cost = np.zeros_like(fut_pref_grid)

        for iteration in range(n_av):
            print('  opt. run:%d/%d'%(iteration+1, n_av))

            # Expand/repeat w_stud for each fut_pref_grid point
            w_stud = np.repeat(w_stud_0.reshape(1,-1), len(fut_pref_grid), axis=0)

            # Shuffle data (assuming sequence is iid in time)
            np.random.shuffle(x_past_whole)
            x_past_sample = x_past_whole[0:n_past_timesteps*batch_size]
            x_buffer_sample = x_past_whole[n_past_timesteps*batch_size:(n_past_timesteps+buffer_size)*batch_size]

            for t in range(n_past_timesteps-1):

                # Batch
                x_batch = x_past_sample[t*batch_size:(t+1)*batch_size]

                # Use only a fraction of samples
                x_batch = x_batch[-n_poisoned_samples:]

                # Perturbation cost
                a = a_greedy_perceptron(w_stud,
                                        w_target,
                                        w_teach,
                                        x_batch,
                                        x_buffer_sample,
                                        dim_input,
                                        eta,
                                        fut_pref_grid*weight_future,
                                        a_min,
                                        a_max,
                                        n_gridpoints,
                                        control_cost_weight,
                                        activation)
                per_cost = (0.5 * control_cost_weight * a**2)

                # Nefarious cost
                phi_s_batch = perceptron(linear(x_batch, w_stud), activation=activation)
                phi_o_batch = perceptron(linear(x_batch, w_target), activation=activation)
                nef_cost = np.mean(0.5*(phi_s_batch-phi_o_batch)**2, axis=0)

                # Total cost
                if t>=n_past_timesteps-window_steadystate:
                    running_cost += per_cost + nef_cost

                # Update
                w_stud = student_update_perceptron(w_stud=w_stud,
                                                   w_teach=w_teach,
                                                   w_target=w_target,
                                                   x_batch=x_batch,
                                                   a=a,
                                                   eta=eta,
                                                   dim_input=dim_input,
                                                   activation=activation,
                                                   actions_students_match=True)

                # Print progress
                interval = int((n_past_timesteps-1)/10)
                if t==0:
                    print('%d/%d'%(t+1, n_past_timesteps-1))
                elif (t+1)%interval==0:
                    print('%d/%d'%(t+1, n_past_timesteps-1))

        # Running cost for fut_pref_val
        running_cost = running_cost/(window_steadystate*n_av)
        running_cost_vs_fut_pref = running_cost
        idx_fut_pref = np.argmin(running_cost)
        fut_pref = fut_pref_grid[idx_fut_pref]
        print('Pre-factor found: %.4f' % fut_pref)

        # Reset initial condition
        w_stud = w_stud_0

    #************************************#
    #       Run perturbed dynamics       #
    #************************************#

    # Time
    n_timesteps = int(x_incoming.shape[0]/batch_size)

    # Arrays to save dynamics
    w_stud_dynamics = np.zeros((n_timesteps, dim_input))
    a_dynamics = np.zeros(n_timesteps)
    nef_cost_dynamics = np.zeros(n_timesteps)
    per_cost_dynamics = np.zeros(n_timesteps)
    cum_cost_dynamics = np.zeros(n_timesteps)
    d_dynamics = np.zeros(n_timesteps)
    acc_dynamics = np.zeros(n_timesteps)

    print('Training...')
    for t in range(n_timesteps):

        # Buffer and batch
        idx_x_buffer = np.random.choice(np.arange(x_past.shape[0]), buffer_size)
        x_buffer = x_past[idx_x_buffer, :]
        x_batch = x_incoming[t*batch_size:(t+1)*batch_size]

        # Fraction of samples to poison
        x_batch_poisoned = x_batch[-n_poisoned_samples:]

        # Action
        a = a_greedy_perceptron(w_stud=w_stud,
                                w_target=w_target,
                                w_teach=w_teach,
                                x_batch=x_batch_poisoned,
                                x_buffer=x_buffer,
                                dim_input=dim_input,
                                eta=eta,
                                weight_future=fut_pref*weight_future,
                                a_min=a_min,
                                a_max=a_max,
                                n_gridpoints=n_gridpoints,
                                control_cost_weight=control_cost_weight,
                                activation=activation)

        # Save current student vector and action
        w_stud_dynamics[t] = w_stud
        a_dynamics[t] = a

        # Save costs
        input_s = np.dot(w_stud, x_batch.T)/(dim_input**0.5)
        input_o = np.dot(w_target, x_batch.T)/(dim_input**0.5)
        label_s = perceptron(input_s, activation=activation)
        label_o = perceptron(input_o, activation=activation)
        nef_cost_dynamics[t] = 0.5 * np.mean((label_s-label_o)**2)
        per_cost_dynamics[t] = 0.5 * control_cost_weight * a**2
        total_cost = nef_cost_dynamics[t] + per_cost_dynamics[t]
        if t==0:
            cum_cost_dynamics[t] = total_cost
        else:
            cum_cost_dynamics[t] = cum_cost_dynamics[t-1] + total_cost * np.exp(-beta*eta*t)

        # Relative distance
        input_s_test = np.dot(w_stud, x_test.T)/(dim_input**0.5)
        label_s_test = perceptron(input_s_test, activation=activation)
        s_pred_test = np.sign(label_s_test)
        d_dynamics[t] = (np.mean((label_s_test-label_t_test)**2)/np.mean((label_o_test-label_t_test)**2))**0.5
        acc_dynamics[t] = np.sum(s_pred_test==t_pred_test.reshape(1,-1), axis=1)/n_samples_test

        # Next student vector
        w_stud = student_update_perceptron(w_stud=w_stud,
                                           w_teach=w_teach,
                                           w_target=w_target,
                                           x_batch=x_batch,
                                           a=a,
                                           eta=eta,
                                           dim_input=dim_input,
                                           activation=activation,
                                           fraction_poisoned_samples=fraction_poisoned_samples,
                                           fraction_poisoned=True)

        # Print progress
        interval = int(n_timesteps/10)
        if t==0:
            print('%d/%d'%(t+1, n_timesteps))
        elif (t+1)%interval==0:
            print('%d/%d'%(t+1, n_timesteps))

    # Results dictionary
    results = {}
    results['w_dynamics'] = w_stud_dynamics
    results['a_dynamics'] = a_dynamics
    results['nef_cost_dynamics'] = nef_cost_dynamics
    results['per_cost_dynamics'] = per_cost_dynamics
    results['cum_cost_dynamics'] = cum_cost_dynamics
    results['d_dynamics'] = d_dynamics
    results['accuracy_dynamics'] = acc_dynamics
    results['running_cost_vs_fut_pref_opt_grid'] = running_cost_vs_fut_pref
    results['fut_pref_opt_grid'] = fut_pref_grid
    results['fut_pref'] = fut_pref

    return results


''' Sample-specific (multi-dimensional) control '''

def exp_greedy_perceptron_multidimcontrol(x_incoming, x_past, x_buffer, x_test, eta, w_teach, w_target, dim_input,
                                          weight_future=[], w_stud_0=[], a_min=0, a_max=1, beta=0.001,
                                          control_cost_weight=1., batch_size=1, n_gridpoints=int(1e2),
                                          buffer_size=250, transient_th=4000, window_steadystate=1000,
                                          activation='Erf', opt_pref=False, fut_pref_min=0.1, fut_pref_max=5.1,
                                          fut_pref_interval=0.1, n_av=10, fut_pref=1., fraction_poisoned_samples=1.):

    # N. poisoned samples
    n_poisoned_samples = int(fraction_poisoned_samples * batch_size)
    n_clean_samples = batch_size - n_poisoned_samples

    # Initialisation
    if len(np.atleast_1d(weight_future))==0:
        weight_future = dim_input/eta
    if not list(w_stud_0):
        w_stud_0 = np.zeros_like(w_teach)
    w_stud = w_stud_0

    # Control cost weight
    control_cost_pref = control_cost_weight
    input_t_test = np.dot(w_teach, x_test.T)/(dim_input**0.5)
    input_o_test = np.dot(w_target, x_test.T)/(dim_input**0.5)
    label_t_test = perceptron(input_t_test, activation=activation)
    label_o_test = perceptron(input_o_test, activation=activation)
    d_target_teach = np.mean((label_o_test-label_t_test)**2)
    control_cost_weight = control_cost_pref * d_target_teach
    print(control_cost_weight)

    # Teach labels
    t_pred_test = np.sign(label_t_test)
    n_samples_test = x_test.shape[0]

    #************************************#
    #       Optimize future weight       #
    #************************************#

    if opt_pref:
        raise ValueError('Future weight optimization not implemented yet for multi-dim control.')

    #************************************#
    #       Run perturbed dynamics       #
    #************************************#

    # Time
    n_timesteps = int(x_incoming.shape[0]/batch_size)

    # Arrays to save dynamics
    w_stud_dynamics = np.zeros((n_timesteps, dim_input))
    a_dynamics = np.zeros((batch_size,n_timesteps))
    nef_cost_dynamics = np.zeros(n_timesteps)
    per_cost_dynamics = np.zeros(n_timesteps)
    cum_cost_dynamics = np.zeros(n_timesteps)
    d_dynamics = np.zeros(n_timesteps)
    acc_dynamics = np.zeros(n_timesteps)

    print('Training...')
    for t in range(n_timesteps):

        # Buffer and batch
        idx_x_buffer = np.random.choice(np.arange(x_past.shape[0]), buffer_size)
        x_buffer = x_past[idx_x_buffer, :]
        x_batch = x_incoming[t*batch_size:(t+1)*batch_size]

        # Fraction of samples to poison
        x_batch_poisoned = x_batch[-n_poisoned_samples:]

        # Action
        a_v = np.zeros(batch_size)
        for idx_sample in range(batch_size):
            a = a_greedy_perceptron(w_stud=w_stud,
                                    w_target=w_target,
                                    w_teach=w_teach,
                                    x_batch=x_batch_poisoned[idx_sample],
                                    x_buffer=x_buffer,
                                    dim_input=dim_input,
                                    eta=eta,
                                    weight_future=fut_pref*weight_future,
                                    a_min=a_min,
                                    a_max=a_max,
                                    n_gridpoints=n_gridpoints,
                                    control_cost_weight=control_cost_weight,
                                    activation=activation)
            a_v[idx_sample] = a

        # Save current student vector and action
        w_stud_dynamics[t] = w_stud
        a_dynamics[:,t] = a_v

        # Save costs
        input_s = np.dot(w_stud, x_batch.T)/(dim_input**0.5)
        input_o = np.dot(w_target, x_batch.T)/(dim_input**0.5)
        label_s = perceptron(input_s, activation=activation)
        label_o = perceptron(input_o, activation=activation)
        nef_cost_dynamics[t] = 0.5 * np.mean((label_s-label_o)**2)
        per_cost_dynamics[t] = np.mean((0.5 * control_cost_weight * a_v**2))
        total_cost = nef_cost_dynamics[t] + per_cost_dynamics[t]
        if t==0:
            cum_cost_dynamics[t] = total_cost
        else:
            cum_cost_dynamics[t] = cum_cost_dynamics[t-1] + total_cost * np.exp(-beta*eta*t)

        # Relative distance and accuracy
        input_s_test = np.dot(w_stud, x_test.T)/(dim_input**0.5)
        label_s_test = perceptron(input_s_test, activation=activation)
        s_pred_test = np.sign(label_s_test)
        d_dynamics[t] = (np.mean((label_s_test-label_t_test)**2)/np.mean((label_o_test-label_t_test)**2))**0.5
        acc_dynamics[t] = np.sum(s_pred_test==t_pred_test.reshape(1,-1), axis=1)/n_samples_test

        # Next student vector
        w_stud_arr = np.zeros((batch_size, dim_input))
        for idx_sample in range(batch_size):
            w_stud_arr[idx_sample,:] = student_update_perceptron(w_stud=w_stud,
                                                                 w_teach=w_teach,
                                                                 w_target=w_target,
                                                                 x_batch=x_batch[idx_sample],
                                                                 a=a_v[idx_sample],
                                                                 eta=eta,
                                                                 dim_input=dim_input,
                                                                 activation=activation,
                                                                 fraction_poisoned_samples=fraction_poisoned_samples,
                                                                 fraction_poisoned=True)
        w_stud = np.mean(w_stud_arr, axis=0)

        # Print progress
        interval = int(n_timesteps/10)
        if t==0:
            print('%d/%d'%(t+1, n_timesteps))
        elif (t+1)%interval==0:
            print('%d/%d'%(t+1, n_timesteps))

    # Results dictionary
    results = {}
    results['w_dynamics'] = w_stud_dynamics
    results['a_dynamics'] = a_dynamics
    results['nef_cost_dynamics'] = nef_cost_dynamics
    results['per_cost_dynamics'] = per_cost_dynamics
    results['cum_cost_dynamics'] = cum_cost_dynamics
    results['d_dynamics'] = d_dynamics
    results['accuracy_dynamics'] = acc_dynamics
    results['fut_pref'] = fut_pref

    return results


# ************* 2-layer NN (NN2L) ***************** #

def exp_greedy_NN2L(x_incoming, x_past, x_buffer, x_test, eta, W_teach, v_teach, W_target, v_target,
                    dim_input, weight_future=[], W_stud_0=[], v_stud_0=[], a_min=0, a_max=1, beta=0.001,
                    control_cost_weight=1., batch_size=1, n_gridpoints=int(1e2), buffer_size=250,
                    transient_th=4000, window_steadystate=1000, activation='Erf', opt_pref=False,
                    fut_pref_min=0.1, fut_pref_max=5.1, fut_pref_interval=0.1, n_av=10, fut_pref=1.,
                    train_first_layer=True, train_second_layer=True, output_scaling='inv_sqroot'):

    # Size hidden layer
    hlayer_size = len(v_teach)

    # Initialisation
    if len(np.atleast_1d(weight_future))==0:
        weight_future = dim_input/eta
    if not list(W_stud_0):
        W_stud_0 = np.zeros_like(W_teach)
    if not list(v_stud_0):
        v_stud_0 = np.zeros_like(v_teach)
    W_stud = W_stud_0
    v_stud = v_stud_0

    # Control cost weight
    control_cost_pref = control_cost_weight
    label_o_test = NN2L(x_test, W_target, v_target,
                        activation=activation,
                        output_scaling=output_scaling)
    label_t_test = NN2L(x_test, W_teach, v_teach,
                        activation=activation,
                        output_scaling=output_scaling)
    d_target_teach = np.mean((label_o_test-label_t_test)**2)
    control_cost_weight = control_cost_pref * d_target_teach
    print('Control cost weight:', control_cost_weight)

    # Cost vs fut_pref
    fut_pref_grid_opt = np.arange(fut_pref_min, fut_pref_max, fut_pref_interval)
    fut_pref_grid = fut_pref * np.ones_like(fut_pref_grid_opt)
    running_cost_vs_fut_pref = np.zeros(len(np.arange(fut_pref_min, fut_pref_max, fut_pref_interval)))

    #************************************#
    #       Optimize future weight       #
    #************************************#

    if opt_pref:
        n_past_timesteps = int(x_past.shape[0]/batch_size)
        upper_bound_pref_fut = fut_pref_min + (round((fut_pref_max-fut_pref_min)/fut_pref_interval)-1)*fut_pref_interval
        print('Optimising future weight pre-factor between %.2f and %.2f'%(fut_pref_min, upper_bound_pref_fut))
        print('N. grid points for future weight: %d' % len(fut_pref_grid_opt))
        print('N past timesteps:', n_past_timesteps)

        x_past_whole = np.concatenate((x_past, x_buffer), axis=0).copy()
        fut_pref_grid = fut_pref_grid_opt
        running_cost = np.zeros_like(fut_pref_grid)

        for iteration in range(n_av):
            print('  opt. run:%d/%d'%(iteration+1, n_av))

            # Expand/repeat W_stud, v_stud for each fut_pref_grid point
            W_stud = np.repeat(np.expand_dims(W_stud_0, axis=0), len(fut_pref_grid), axis=0)
            v_stud = np.repeat(np.expand_dims(v_stud_0, axis=0), len(fut_pref_grid), axis=0)

            # Shuffle data (assuming sequence is iid in time)
            np.random.shuffle(x_past_whole)
            x_past_sample = x_past_whole[0:n_past_timesteps*batch_size]
            x_buffer_sample = x_past_whole[n_past_timesteps*batch_size:(n_past_timesteps+buffer_size)*batch_size]

            for t in range(n_past_timesteps-1):

                # Batch
                x_batch = x_past_sample[t*batch_size:(t+1)*batch_size]

                # Perturbation cost
                a = a_greedy_NN2L(W_stud=W_stud,
                                  v_stud=v_stud,
                                  W_target=W_target,
                                  v_target=v_target,
                                  W_teach=W_teach,
                                  v_teach=v_teach,
                                  x_batch=x_batch,
                                  x_buffer=x_buffer_sample,
                                  dim_input=dim_input,
                                  eta=eta,
                                  weight_future=fut_pref_grid*weight_future,
                                  a_min=a_min,
                                  a_max=a_max,
                                  n_gridpoints=n_gridpoints,
                                  control_cost_weight=control_cost_weight,
                                  activation=activation,
                                  train_first_layer=train_first_layer,
                                  train_second_layer=train_second_layer,
                                  output_scaling=output_scaling)
                per_cost = 0.5 * control_cost_weight * a**2

                # Nefarious cost
                phi_s_batch = NN2L(x_batch, W_stud, v_stud,
                                   activation=activation,
                                   output_scaling=output_scaling)
                phi_o_batch = NN2L(x_batch, W_target, v_target,
                                   activation=activation,
                                   output_scaling=output_scaling)
                nef_cost = np.mean(0.5*(phi_s_batch-phi_o_batch)**2, axis=0)

                # Total cost
                if t>=n_past_timesteps-window_steadystate:
                    running_cost += per_cost + nef_cost

                # Update
                W_stud, v_stud = student_update_NN2L(W_stud=W_stud,
                                                       v_stud=v_stud,
                                                       W_target=W_target,
                                                       v_target=v_target,
                                                       W_teach=W_teach,
                                                       v_teach=v_teach,
                                                       x_batch=x_batch,
                                                       a=a,
                                                       eta=eta,
                                                       dim_input=dim_input,
                                                       activation=activation,
                                                       actions_students_match=True,
                                                       train_first_layer=train_first_layer,
                                                       train_second_layer=train_second_layer,
                                                       output_scaling=output_scaling)

                # Print progress
                interval = int((n_past_timesteps-1)/10)
                if t==0:
                    print('%d/%d'%(t+1, n_past_timesteps-1))
                elif (t+1)%interval==0:
                    print('%d/%d'%(t+1, n_past_timesteps-1))

        # Running cost for fut_pref_val
        running_cost = running_cost/(window_steadystate*n_av)

        running_cost_vs_fut_pref = running_cost
        idx_fut_pref = np.argmin(running_cost)
        fut_pref = fut_pref_grid[idx_fut_pref]
        print('Pre-factor found: %.4f' % fut_pref)

        # Reset initial condition
        W_stud = W_stud_0
        v_stud = v_stud_0

    #************************************#
    #       Run perturbed dynamics       #
    #************************************#

    # Time
    n_timesteps = int(x_incoming.shape[0]/batch_size)

    # Arrays to save dynamics
    v_stud_dynamics = np.zeros((n_timesteps, hlayer_size))
    a_dynamics = np.zeros(n_timesteps)
    nef_cost_dynamics = np.zeros(n_timesteps)
    per_cost_dynamics = np.zeros(n_timesteps)
    cum_cost_dynamics = np.zeros(n_timesteps)
    d_dynamics = np.zeros(n_timesteps)

    print('Training...')
    for t in range(n_timesteps):

        # Buffer and batch
        idx_x_buffer = np.random.choice(np.arange(x_past.shape[0]), buffer_size)
        x_buffer = x_past[idx_x_buffer, :]
        x_batch = x_incoming[t*batch_size:(t+1)*batch_size]

        # Action
        a = a_greedy_NN2L(W_stud=W_stud,
                          v_stud=v_stud,
                          W_target=W_target,
                          v_target=v_target,
                          W_teach=W_teach,
                          v_teach=v_teach,
                          x_batch=x_batch,
                          x_buffer=x_buffer,
                          dim_input=dim_input,
                          eta=eta,
                          weight_future=fut_pref*weight_future,
                          a_min=a_min,
                          a_max=a_max,
                          n_gridpoints=n_gridpoints,
                          control_cost_weight=control_cost_weight,
                          activation=activation,
                          train_first_layer=train_first_layer,
                          train_second_layer=train_second_layer,
                          output_scaling=output_scaling)

        # Save current student parameters and action
        v_stud_dynamics[t] = v_stud
        a_dynamics[t] = a

        # Costs
        label_s = NN2L(x_batch, W_stud, v_stud,
                       activation=activation,
                       output_scaling=output_scaling)
        label_o = NN2L(x_batch, W_target, v_target,
                       activation=activation,
                       output_scaling=output_scaling)
        nef_cost_dynamics[t] = 0.5 * np.mean((label_s-label_o)**2)
        per_cost_dynamics[t] = 0.5 * control_cost_weight * a**2
        total_cost = nef_cost_dynamics[t] + per_cost_dynamics[t]
        if t==0:
            cum_cost_dynamics[t] = total_cost
        else:
            cum_cost_dynamics[t] = cum_cost_dynamics[t-1] + total_cost * np.exp(-beta*eta*t)

        # Relative distance
        label_s_test = NN2L(x_test, W_stud, v_stud,
                            activation=activation,
                            output_scaling=output_scaling)
        d_dynamics[t] = (np.mean((label_s_test-label_t_test)**2)/np.mean((label_o_test-label_t_test)**2))**0.5

        # Next student weights
        W_stud, v_stud = student_update_NN2L(W_stud=W_stud,
                                               v_stud=v_stud,
                                               W_target=W_target,
                                               v_target=v_target,
                                               W_teach=W_teach,
                                               v_teach=v_teach,
                                               x_batch=x_batch,
                                               a=a,
                                               eta=eta,
                                               dim_input=dim_input,
                                               activation=activation,
                                               train_first_layer=train_first_layer,
                                               train_second_layer=train_second_layer,
                                               output_scaling=output_scaling)

        # Print progress
        interval = int(n_timesteps/10)
        if t==0:
            print('%d/%d'%(t+1, n_timesteps))
        elif (t+1)%interval==0:
            print('%d/%d'%(t+1, n_timesteps))

    # Results dictionary
    results = {}
    results['v_dynamics'] = v_dynamics
    results['a_dynamics'] = a_dynamics
    results['nef_cost_dynamics'] = nef_cost_dynamics
    results['per_cost_dynamics'] = per_cost_dynamics
    results['cum_cost_dynamics'] = cum_cost_dynamics
    results['d_dynamics'] = d_dynamics
    results['running_cost_vs_fut_pref_opt_grid'] = running_cost_vs_fut_pref
    results['fut_pref_opt_grid'] = fut_pref_grid
    results['fut_pref'] = fut_pref

    return results
