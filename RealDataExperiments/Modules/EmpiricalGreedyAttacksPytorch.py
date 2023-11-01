## Import statements ##
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy


# ************************************************ #
#                                                  #
#                 Model classes                    #
#                                                  #
# ************************************************ #

''' Learner class: perceptron '''

class PerceptronModel(torch.nn.Module):

    def __init__(self, input_size, activation, parameters=[]):
        super(PerceptronModel, self).__init__()

        self.activation = activation
        self.input_size = input_size
        self.linear1 = torch.nn.Linear(input_size, 1, bias=False)

        if len(parameters)>0:
            self.linear1.weight = torch.nn.Parameter(parameters)

    def forward(self, x):
        x = self.linear1(x)/self.input_size**0.5
        if self.activation=='Linear':
            pass
        elif self.activation=='Erf':
            x = torch.erf(x/2**0.5)
        elif self.activation=='ReLU':
            x = torch.ReLU(x)
        elif self.activation=='Tanh':
            x = torch.Tanh(x)
        return x

    def forward_preactivation(self, x):
        x = self.linear1(x)/self.input_size**0.5
        return x

    def predict(self, x):
        return torch.sign(self.forward(x))


# ************************************************ #

''' Learner class: LeNet '''

class LeNet(nn.Module):

    def __init__(self, n_channels_input=1):
        super(LeNet, self).__init__()
        # conv. layers
        self.conv1 = nn.Conv2d(n_channels_input, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # full layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) # self.fc3(x) # For linear output activation
        return x

    def predict(self, x):
        return torch.sign(self.forward(x))

class LeNetLinLast(nn.Module):

    def __init__(self, n_channels_input=1):
        super(LeNet, self).__init__()
        # conv. layers
        self.conv1 = nn.Conv2d(n_channels_input, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # full layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) # self.fc3(x) # For linear output activation
        return x

    def predict(self, x):
        return torch.sign(self.forward(x))


# ************************************************ #

''' Learner class: ResNet '''

class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1,
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        self.downsample = downsample

    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            i = self.downsample(i)
        x += i
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, config, inputlastfc_dim=10, dobatchnorm=True):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, inputlastfc_dim)
        self.lastbn = nn.BatchNorm1d(inputlastfc_dim)
        self.dobatchnorm = dobatchnorm
        self.lastfc = nn.Linear(inputlastfc_dim, 1, bias=False)

    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):

        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))
        self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)

    def forward_prelastfc(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        # last fc
        x = F.relu(x)
        if self.dobatchnorm:
            x = self.lastbn(x)

        return x

    def forward_lastfc(self, x):

        x = self.lastfc(self.forward_prelastfc(x))

        return x

    def forward(self, x):

        x = torch.erf(self.forward_lastfc(x)/2**0.5)

        return x


# ************************************************ #

''' Learner class: VGG '''

class VGG(nn.Module):
    def __init__(self, features, denselayers_width=4096, inputlastfc_dim=10, dobatchnorm=True):
        super().__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, denselayers_width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(denselayers_width, denselayers_width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(denselayers_width, inputlastfc_dim),
        )
        self.lastbn = nn.BatchNorm1d(inputlastfc_dim)
        self.dobatchnorm = dobatchnorm
        self.lastfc = nn.Linear(inputlastfc_dim, 1, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        # last fc
        x = F.relu(x)
        if self.dobatchnorm:
            x = self.lastbn(x)
        x = self.lastfc(x)
        x = torch.erf(x/2**0.5)

        return x #, h

    def forward_lastfc(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        # last fc
        x = F.relu(x)
        if self.dobatchnorm:
            x = self.lastbn(x)
        x = self.lastfc(x)

        return x #, h

    def forward_prelastfc(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        # last fc
        x = F.relu(x)
        if self.dobatchnorm:
            x = self.lastbn(x)

        return x #, h

def get_vgg_layers(config, batch_norm):

    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)



# ************************************************ #
#                                                  #
#                  Greedy policy                   #
#                                                  #
# ************************************************ #

''' Greedy attacks given model and minibatch '''

def a_greedy(model, optimizer, minibatch, test_examples, control_cost_weight,
             future_weight, lr, a_min, a_max, n_gridpoints=100, optimizer_type='SGD'):

    # Test examples
    x_test, y_test, y_target_test = test_examples

    # Criterion
    criterion = torch.nn.MSELoss()

    # Input and output
    x, y, y_target = minibatch

    # a grid
    a_grid = np.linspace(a_min, a_max, n_gridpoints)
    two_step_cost_arr = np.zeros_like(a_grid)

    for a_idx, a in enumerate(a_grid):

        # Perturbation cost
        per_cost = 0.5 * control_cost_weight * a**2

        # Perturbed labels
        y_perturbed = y*(1-a) + y_target*a

        # Update
        model_copy = copy.deepcopy(model)
        if optimizer_type=='SGD':
            optimizer_copy = torch.optim.SGD(model_copy.parameters(), lr=lr)
        elif optimizer_type=='Adam':
            optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=lr)
        else:
            raise ValueError('Optimizer type must be SGD or Adam.')
        optimizer_copy.load_state_dict(optimizer.state_dict())
        y_pred_copy = model_copy(x)
        loss_copy = 0.5 * criterion(y_pred_copy.reshape(-1,1), y_perturbed.reshape(-1,1))
        optimizer_copy.zero_grad()
        loss_copy.backward()
        optimizer_copy.step()

        # Nefarious cost
        with torch.no_grad():
            y_pred_test_copy = model_copy(x_test)
            nef_cost = 0.5 * future_weight * criterion(y_pred_test_copy.reshape(-1,1), y_target_test.reshape(-1,1))

        # Total running cost
        two_step_cost_arr[a_idx] = per_cost + nef_cost

    # Chosen action
    a = a_grid[np.argmin(two_step_cost_arr)]

    return a



# ************************************************ #
#                                                  #
#            Labelled data experiment              #
#                                                  #
# ************************************************ #

''' Experiment greedy attack '''

def labelleddata_exp_greedy(model_type, dataset, n_timesteps, n_past_timesteps, eta, dim_input=10, n_channels=1,
                            weight_future=[], a_min=0, a_max=1, n_gridpoints=100, beta=0.001, control_cost_pref=1.,
                            batch_size=1, buffer_size=1000, test_size=1000, transient_th=10000, window_steadystate=1000,
                            activation='Erf', opt_pref=False, fut_pref_min=0.1, fut_pref_max=5.1, fut_pref_interval=0.1,
                            fut_pref=1., momentum=0., w_regularizer=0, attacker_on=True, export_while_running=False,
                            export_results=False, export_path=None, exp_description=None, use_cuda_if_available=True,
                            teacher_smoothlabels=False, optimizer_type='SGD', fraction_poisoned_samples=1.):

    '''
       1. This function runs greedy label-flipping attacks on a binary classification
       problem: labels are either +1 or -1, and the target has the signs flipped.
       If teacher_smoothlabels==TRUE, teacher labels are provided by a function
       trained on clean data.

       2. data_* assumed to be of the form TensorDataset(x_incoming, y_incoming).
       x_incoming according to model types:
           - Perceptron: takes as input tensors of shape (n_samples, dim_input)
           - LeNet, LeNetLinLast: take as input tensors of shape (batch_size, n_channels, 32, 32)
    '''

    # Set Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not use_cuda_if_available:
        device = 'cpu'
    print('Device:', device)

    # N. poisoned samples
    n_poisoned_samples = int(fraction_poisoned_samples * batch_size)
    n_clean_samples = batch_size - n_poisoned_samples

    # Models initialization
    if model_type=='Perceptron':
        model_stud = PerceptronModel(dim_input, activation)
    elif model_type=='LeNet':
        model_stud = LeNet(n_channels)
    elif model_type=='LeNetLinLast':
        model_stud = LeNetLinLast(n_channels)
    model_stud.to(device)

    # Weight future
    if len(np.atleast_1d(weight_future))==0:
        if model_type in ['LeNet', 'LeNetLinLast']:
            weight_future = 1/eta
        else:
            weight_future = dim_input/eta
    print('Base future weight:', weight_future)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    if optimizer_type=='SGD':
        optimizer = torch.optim.SGD(model_stud.parameters(), lr=eta, momentum=momentum, weight_decay=w_regularizer)
    elif optimizer_type=='Adam':
        optimizer = torch.optim.Adam(model_stud.parameters(), lr=eta, weight_decay=w_regularizer)
    else:
        raise ValueError('Optimizer type must be SGD or Adam.')

    # Datasets and loaders
    # --- incoming --- #
    data_loader_incoming = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    iter_data_incoming = iter(data_loader_incoming)
    len_batch = len(data_loader_incoming)
    # --- past --- #
    data_loader_past = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    iter_data_past = iter(data_loader_past)
    # --- buffer --- #
    data_loader_buffer = DataLoader(dataset, batch_size=buffer_size, shuffle=True)
    data_buffer = next(iter(data_loader_buffer))
    x_buffer, y_buffer = data_buffer
    x_buffer, y_buffer = x_buffer.to(device), y_buffer.to(device)
    y_target_buffer = -y_buffer.detach().clone().to(device)
    data_buffer = (x_buffer, y_buffer, y_target_buffer)
    # --- test --- #
    data_loader_test = DataLoader(dataset, batch_size=test_size, shuffle=True)
    data_test = next(iter(data_loader_test))
    x_test, y_test = data_test
    x_test, y_test = x_test.to(device), y_test.to(device)
    y_target_test = -y_test.detach().clone().to(device)
    labels_test = y_test.detach().clone().to(device)
    data_test = (x_test, y_test, y_target_test)
    loss_target_test = 0.5 * criterion(y_target_test.double().reshape(-1,1), y_test.double().reshape(-1,1)) #
    print('loss_target_test', loss_target_test)

    # Export directory
    if export_results:
        filename_example = 'quantity' + '__@@@__' + exp_description
        export_dir_example = export_path + filename_example
        print('Exporting files to:\n', export_dir_example)

    # Control cost
    control_cost_weight = control_cost_pref * (2*loss_target_test)
    print('control_cost_weight:', control_cost_weight)

    # Results dictionary
    results = {}

    # Initialize arrays dynamics
    #w_stud_dynamics = np.zeros((n_timesteps, dim_input))
    d_dynamics_clean = np.zeros(n_past_timesteps)
    accuracy_dynamics_clean = np.zeros(n_past_timesteps)


    #******************************#
    #                              #
    #      Run clean dynamics      #
    #                              #
    #******************************#

    interval = int(n_past_timesteps/10) # 10
    print('Clean training')
    for t in range(n_past_timesteps):

        # Reset iterator
        if t>=len_batch:
            iter_data_past = iter(data_loader_past)

        # Batch
        minibatch = next(iter_data_past)
        x, y = minibatch
        x, y = x.to(device), y.to(device)
        y_target = -y.detach().clone().to(device)

        # Save test distance and accuracy
        with torch.no_grad():
            y_pred_test = model_stud(x_test)
            loss_test = 0.5 * criterion(y_pred_test.reshape(-1,1), y_test.reshape(-1,1))
            d = loss_test.item() / loss_target_test.item()
            d_dynamics_clean[t] = d**0.5
            class_pred_test = model_stud.predict(x_test)
            accuracy_dynamics_clean[t] = torch.sum(class_pred_test.reshape(-1,)==labels_test.reshape(-1,))/len(y_test)

        # Model predictions on current batch
        y_pred = model_stud(x)

        # Action set to zero
        a = float(0)

        # Update model
        y_perturbed = y*(1-a) + y_target*a
        loss = 0.5 * criterion(y_pred.reshape(-1,1), y_perturbed.reshape(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Export results
        if export_while_running:
            if t%500==0:
                results['accuracy_dynamics_clean'] = accuracy_dynamics_clean
                results['d_dynamics_clean'] = d_dynamics_clean
                results['time_progress_clean'] = np.arange(t)

                for name in results.keys():
                    data_to_export = results[name]
                    filename = name + '__@@@__' + exp_description
                    np.save(export_path + filename, data_to_export)

        # print progression
        if t==0:
            print('%d/%d'%(t+1, n_past_timesteps))
        elif (t+1)%interval==0:
            print('%d/%d'%(t+1, n_past_timesteps))

    # Results dictionary
    results['accuracy_dynamics_clean'] = accuracy_dynamics_clean
    results['d_dynamics_clean'] = d_dynamics_clean
    results['time_progress_clean'] = np.arange(t)

    # Re-scale learned weights (Perceptron)
    if model_type=='Perceptron':
        w_teach = model_stud.linear1.weight.data.clone().detach().reshape(-1,)
        w_teach = w_teach /(np.dot(w_teach, w_teach)/dim_input)**0.5
        model_stud = PerceptronModel(dim_input, activation, parameters=w_teach)
        model_stud.to(device)
        if optimizer_type=='SGD':
            optimizer = torch.optim.SGD(model_stud.parameters(), lr=eta, momentum=momentum, weight_decay=w_regularizer)
        elif optimizer_type=='Adam':
            optimizer = torch.optim.Adam(model_stud.parameters(), lr=eta, weight_decay=w_regularizer)
        else:
            raise ValueError('Optimizer type must be SGD or Adam.')

    #******************************#
    #       Teacher function       #
    #******************************#

    if teacher_smoothlabels:
        # Copy clean model into teacher
        model_teach = copy.deepcopy(model_stud)
        model_teach.to(device)

        # Re-set clean test labels
        y_test = model_teach(x_test).detach().clone().to(device)
        labels_test = np.sign(y_test.cpu()).detach().clone().to(device)
        y_target_test = -y_test.detach().clone().to(device)
        data_test = (x_test, y_test, y_target_test)
        loss_target_test = 0.5 * criterion(y_target_test.double().reshape(-1,1), y_test.double().reshape(-1,1))
        control_cost_weight = control_cost_pref * (2*loss_target_test)
        print('loss_target_test', loss_target_test)
        print('control_cost_weight', control_cost_weight)

        # Re-set buffer labels
        y_buffer = model_teach(x_buffer).detach().clone().to(device)
        y_target_buffer = -y_buffer.detach().clone().to(device)
        data_buffer = (x_buffer, y_buffer, y_target_buffer)

        # Re-initialize student
        if model_type=='Perceptron':
            model_stud = PerceptronModel(dim_input, activation)
        elif model_type=='LeNet':
            model_stud = LeNet(n_channels)
        elif model_type=='LeNetLinLast':
            model_stud = LeNetLinLast(n_channels)
        model_stud.to(device)

        # Loss and optimizer
        if optimizer_type=='SGD':
            optimizer = torch.optim.SGD(model_stud.parameters(), lr=eta, momentum=momentum, weight_decay=w_regularizer)
        elif optimizer_type=='Adam':
            optimizer = torch.optim.Adam(model_stud.parameters(), lr=eta, weight_decay=w_regularizer)
        else:
            raise ValueError('Optimizer type must be SGD or Adam.')

        # Run clean dynamics again
        print('Clean training, teacher labels')

        # Reset iterator
        iter_data_past = iter(data_loader_past)

        for t in range(n_past_timesteps):

            # Reset iterator
            if t>=len_batch:
                iter_data_past = iter(data_loader_past)

            # Batch
            minibatch = next(iter_data_past)
            x, y = minibatch
            x = x.to(device)
            y = model_teach(x)
            y = y.detach().clone()
            x, y = x.to(device), y.to(device)
            y_target = -y.detach().clone().to(device)

            # Save test distance and accuracy
            with torch.no_grad():
                y_pred_test = model_stud(x_test)
                loss_test = 0.5 * criterion(y_pred_test.reshape(-1,1), y_test.reshape(-1,1))
                d = loss_test.item() / loss_target_test.item()
                d_dynamics_clean[t] = d**0.5
                class_pred_test = model_stud.predict(x_test)
                accuracy_dynamics_clean[t] = torch.sum(class_pred_test.reshape(-1,)==labels_test.reshape(-1,))/len(y_test)

            # Model predictions on current batch
            y_pred = model_stud(x)

            # Action set to zero
            a = float(0)

            # Update model
            y_perturbed = y*(1-a) + y_target*a
            loss = 0.5 * criterion(y_pred.reshape(-1,1), y_perturbed.reshape(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Export results
            if export_while_running:
                if t%500==0:
                    results['accuracy_dynamics_clean'] = accuracy_dynamics_clean
                    results['d_dynamics_clean'] = d_dynamics_clean
                    results['time_progress_clean'] = np.arange(t)

                    for name in results.keys():
                        data_to_export = results[name]
                        filename = name + '__@@@__' + exp_description
                        np.save(export_path + filename, data_to_export)

            # Print progression
            if t==0:
                print('%d/%d'%(t+1, n_past_timesteps))
            elif (t+1)%interval==0:
                print('%d/%d'%(t+1, n_past_timesteps))

        # Results dictionary
        results['accuracy_dynamics_clean'] = accuracy_dynamics_clean
        results['d_dynamics_clean'] = d_dynamics_clean
        results['time_progress_clean'] = np.arange(t)

    # Export results clean
    if export_results:
        for name in results.keys():
            data_to_export = results[name]
            filename = name + '__@@@__' + exp_description
            np.save(export_path + filename, data_to_export)



    #**********************************#
    #                                  #
    #      Run perturbed dynamics      #
    #                                  #
    #**********************************#

    # Cost vs fut_pref
    fut_pref_grid_opt = np.arange(fut_pref_min, fut_pref_max, fut_pref_interval)
    fut_pref_grid = fut_pref * np.ones_like(fut_pref_grid_opt)
    running_cost_vs_fut_pref = np.zeros(len(np.arange(fut_pref_min, fut_pref_max, fut_pref_interval)))

    if attacker_on:

        #************************************#
        #       Optimize future weight       #
        #************************************#

        if opt_pref:
            upper_bound_pref_fut = max(fut_pref_grid_opt)
            print('Optimising future weight pre-factor between %.2f and %.2f'%(fut_pref_min, upper_bound_pref_fut))
            print('N. grid points for future weight: %d' % len(fut_pref_grid_opt))
            print('N past timesteps:', n_past_timesteps)
            fut_pref_grid = fut_pref_grid_opt

            for idx_fut_pref, fut_pref_val in enumerate(fut_pref_grid_opt):
                print('%d/%d' % (idx_fut_pref+1, len(fut_pref_grid_opt)))

                # Reset iterator
                iter_data_past = iter(data_loader_past)

                # Copy model and optimizer
                model_stud_copy = copy.deepcopy(model_stud)
                if optimizer_type=='SGD':
                    optimizer_copy = torch.optim.SGD(model_stud_copy.parameters(), lr=eta, momentum=momentum, weight_decay=w_regularizer)
                elif optimizer_type=='Adam':
                    optimizer_copy = torch.optim.Adam(model_stud_copy.parameters(), lr=eta, weight_decay=w_regularizer)
                else:
                    raise ValueError('Optimizer type must be SGD or Adam.')
                optimizer_copy.load_state_dict(optimizer.state_dict()) # copy state

                # Run dynamics using past data <- improvable: running multiple dynamics
                av_running_cost_ss = 0
                for t in range(n_past_timesteps):

                    # Reset iterator
                    if t>=len_batch:
                        iter_data_past = iter(data_loader_past)

                    # Batch
                    minibatch = next(iter_data_past)

                    # Use only a fraction of samples
                    x, y = minibatch
                    x, y = x[-n_poisoned_samples:], y[-n_poisoned_samples:]

                    # Teacher smoothing
                    if teacher_smoothlabels:
                        x = x.to(device)
                        y = model_teach(x)
                        y = y.detach().clone()
                    x, y = x.to(device), y.to(device)
                    y_target = -y.detach().clone().to(device)
                    minibatch = (x, y, y_target)

                    # Action
                    a = a_greedy(model_stud_copy, optimizer_copy, minibatch, data_buffer, control_cost_weight,
                                 future_weight=fut_pref_val*weight_future, lr=eta,
                                 a_min=a_min, a_max=a_max, n_gridpoints=n_gridpoints, optimizer_type=optimizer_type)

                    # Model predictions on current batch
                    y_pred = model_stud_copy(x)

                    # Compute costs
                    with torch.no_grad():
                        nef_cost = 0.5 * criterion(y_pred.reshape(-1,1), y_target.reshape(-1,1))
                        per_cost = 0.5 * control_cost_weight * a**2
                        running_cost = nef_cost + per_cost
                        if t>=n_past_timesteps-window_steadystate:
                            av_running_cost_ss += running_cost

                    # Update model
                    y_perturbed = y*(1-a) + y_target*a
                    y_perturbed[0:n_clean_samples] = y[0:n_clean_samples]
                    loss = 0.5 * criterion(y_pred.reshape(-1,1), y_perturbed.reshape(-1,1))
                    optimizer_copy.zero_grad()
                    loss.backward()
                    optimizer_copy.step()

                # Running cost for fut_pref_val
                av_running_cost_ss = av_running_cost_ss/window_steadystate
                running_cost_vs_fut_pref[idx_fut_pref] = av_running_cost_ss

            # Best fut_pref_val
            idx_fut_pref = np.argmin(running_cost_vs_fut_pref)
            fut_pref = fut_pref_grid_opt[idx_fut_pref]
            print('fut_pref found:', fut_pref)

        #******************************#
        #         Run dynamics         #
        #******************************#

        # Initialize arrays dynamics
        #w_stud_dynamics = np.zeros((n_timesteps, dim_input))
        a_dynamics = np.zeros(n_timesteps)
        nef_cost_dynamics = np.zeros(n_timesteps)
        per_cost_dynamics = np.zeros(n_timesteps)
        cum_cost_dynamics = np.zeros(n_timesteps)
        d_dynamics = np.zeros(n_timesteps)
        accuracy_dynamics = np.zeros(n_timesteps)

        # Reset iterator
        iter_data_past = iter(data_loader_past)

        interval = int(n_timesteps/10) # 10
        print('Poisoned training')
        for t in range(n_timesteps):

            # Reset iterator
            if t>=len_batch:
                iter_data_incoming = iter(data_loader_incoming)

            # Batch
            minibatch = next(iter_data_incoming)
            x, y = minibatch
            if teacher_smoothlabels:
                x = x.to(device)
                y = model_teach(x)
                y = y.detach().clone()
            x, y = x.to(device), y.to(device)
            y_target = -y.detach().clone().to(device)

            # Fraction of samples to poison
            minibatch = (x[-n_poisoned_samples:],
                         y[-n_poisoned_samples:],
                         y_target[-n_poisoned_samples:])

            # Action
            a = a_greedy(model_stud, optimizer, minibatch, data_buffer, control_cost_weight,
                         future_weight=fut_pref*weight_future, lr=eta,
                         a_min=a_min, a_max=a_max, n_gridpoints=n_gridpoints, optimizer_type=optimizer_type)

            # Save current student vector and action
            #w_stud_dynamics[t] = model_stud.linear1.weight.detach().numpy()
            a_dynamics[t] = a

            # Model predictions on current batch
            y_pred = model_stud(x)

            # Save costs, test distance and accuracy
            with torch.no_grad():

                # Costs
                nef_cost_dynamics[t] = 0.5 * criterion(y_pred.reshape(-1,1), y_target.reshape(-1,1))
                per_cost_dynamics[t] = 0.5 * control_cost_weight * a**2
                running_cost = nef_cost_dynamics[t] + per_cost_dynamics[t]
                if t==0:
                    cum_cost_dynamics[t] = running_cost
                else:
                    cum_cost_dynamics[t] = cum_cost_dynamics[t-1] + running_cost * np.exp(-beta*eta*t)

                # Relative distance and accuracy
                y_pred_test = model_stud(x_test)
                loss_test = 0.5 * criterion(y_pred_test.reshape(-1,1), y_test.reshape(-1,1))
                d = loss_test.item() / loss_target_test.item()
                d_dynamics[t] = d**0.5
                class_pred_test = model_stud.predict(x_test)
                accuracy_dynamics[t] = torch.sum(class_pred_test.reshape(-1,)==labels_test.reshape(-1,))/len(y_test)

            # Update model
            y_perturbed = y*(1-a) + y_target*a
            y_perturbed[0:n_clean_samples] = y[0:n_clean_samples]
            loss = 0.5 * criterion(y_pred.reshape(-1,1), y_perturbed.reshape(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progression
            if t==0:
                print('%d/%d'%(t+1, n_timesteps))
            elif (t+1)%interval==0:
                print('%d/%d'%(t+1, n_timesteps))

            # Export results
            if export_while_running:
                if t%500==0:
                    #results['w_dynamics'] = w_stud_dynamics
                    results['a_dynamics'] = a_dynamics
                    results['nef_cost_dynamics'] = nef_cost_dynamics
                    results['per_cost_dynamics'] = per_cost_dynamics
                    results['cum_cost_dynamics'] = cum_cost_dynamics
                    results['d_dynamics'] = d_dynamics
                    results['running_cost_vs_fut_pref_opt_grid'] = running_cost_vs_fut_pref
                    results['fut_pref_opt_grid'] = fut_pref_grid
                    results['fut_pref'] = fut_pref
                    results['accuracy_dynamics'] = accuracy_dynamics
                    results['time_progress'] = np.arange(t)

                    for name in results.keys():
                        data_to_export = results[name]
                        filename = name + '__@@@__' + exp_description
                        np.save(export_path + filename, data_to_export)

        print('Perturbed training completed!')

        # Results dictionary
        #results['w_dynamics'] = w_stud_dynamics
        results['a_dynamics'] = a_dynamics
        results['nef_cost_dynamics'] = nef_cost_dynamics
        results['per_cost_dynamics'] = per_cost_dynamics
        results['cum_cost_dynamics'] = cum_cost_dynamics
        results['d_dynamics'] = d_dynamics
        results['running_cost_vs_fut_pref_opt_grid'] = running_cost_vs_fut_pref
        results['fut_pref_opt_grid'] = fut_pref_grid
        results['fut_pref'] = fut_pref
        results['accuracy_dynamics'] = accuracy_dynamics
        results['time_progress'] = np.arange(t)

        # Export results:
        if export_results:
            for name in results.keys():
                data_to_export = results[name]
                filename = name + '__@@@__' + exp_description
                np.save(export_path + filename, data_to_export)

    return results


# ************************************************************* #

''' Experiment greedy attack - transfer learning (perceptron) '''

def labelleddata_exp_greedy_ErfTL(trainset, testset, w_teach, n_timesteps, n_past_timesteps, eta, dim_input,
                                  weight_future=[], w_stud_0=[], a_min=0, a_max=1, beta=0.001, control_cost_pref=1.,
                                  batch_size=1, n_gridpoints=int(1e3), buffer_size=250, transient_th=10000,
                                  window_steadystate=1000, activation='Erf', opt_pref=False, fut_pref_min=0.1,
                                  fut_pref_max=5.1, fut_pref_interval=0.1, fut_pref=1., momentum=0,
                                  w_regularizer=0, use_cuda_if_available=True, export_results=False,
                                  export_while_running=False, export_path=None, exp_description=None):

    # Set Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not use_cuda_if_available:
        device = 'cpu'
    print('Device:', device)

    # Weight future
    if len(np.atleast_1d(weight_future))==0:
        weight_future = 1/eta

    # Define models
    w_teach_tensor = torch.tensor(w_teach)
    teacher_model = PerceptronModel(input_size=dim_input, activation=activation, parameters=w_teach_tensor).to(device)
    if len(w_stud_0)>0:
        w_stud_tensor = torch.tensor(w_stud_0)
    else:
        w_stud_tensor = torch.normal(0, 1, (dim_input,))
    student_model = PerceptronModel(input_size=dim_input, activation=activation, parameters=w_stud_tensor).to(device)

    # Re-set labels if using teacher
    trainset_input = trainset[:][0]
    trainset_output = teacher_model(trainset_input.to(device)).detach().clone()
    trainset = TensorDataset(trainset_input,trainset_output)
    testset_input = testset[:][0]
    testset_output = teacher_model(testset_input.to(device)).detach().clone()
    testset = TensorDataset(testset_input,testset_output)

    # Test labels
    labels_test = torch.sign(testset[:][1]).detach().clone().to(device)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=eta, momentum=momentum, weight_decay=w_regularizer)

    # Datasets and loaders
    # --- incoming --- #
    data_loader_incoming = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    iter_data_incoming = iter(data_loader_incoming)
    len_batch = len(data_loader_incoming)
    # --- past --- #
    data_loader_past = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    iter_data_past = iter(data_loader_past)
    # --- buffer --- #
    data_loader_buffer = DataLoader(trainset, batch_size=buffer_size, shuffle=True)
    data_buffer = next(iter(data_loader_buffer))
    x_buffer, y_buffer = data_buffer
    x_buffer, y_buffer = x_buffer.to(device), y_buffer.to(device)
    y_target_buffer = -y_buffer.detach().clone().to(device)
    data_buffer = (x_buffer, y_buffer, y_target_buffer)
    # --- test --- #
    x_test, y_test = testset[:][0], testset[:][1]
    x_test, y_test = x_test.to(device), y_test.to(device)
    y_target_test = -y_test.detach().clone().to(device)
    data_test = (x_test, y_test, y_target_test)

    # Compute loss target
    loss_target_test = 0.5 * criterion(y_target_test.double().reshape(-1,1), y_test.double().reshape(-1,1)) #
    print('loss_target_test', loss_target_test)

    # Export directory
    if export_results:
        filename_example = 'quantity' + '__@@@__' + exp_description
        export_dir_example = export_path + filename_example
        print('Exporting files to:\n', export_dir_example)

    # Control cost
    control_cost_weight = control_cost_pref * (2*loss_target_test)
    print('control_cost_weight:', control_cost_weight)

    # Results dictionary
    results = {}

    # Cost vs fut_pref
    fut_pref_grid_opt = np.arange(fut_pref_min, fut_pref_max, fut_pref_interval)
    fut_pref_grid = fut_pref * np.ones_like(fut_pref_grid_opt)
    running_cost_vs_fut_pref = np.zeros(len(fut_pref_grid_opt))

    #************************************#
    #       Optimize future weight       #
    #************************************#

    if opt_pref:
        upper_bound_pref_fut = max(fut_pref_grid_opt)
        print('Optimising future weight pre-factor between %.2f and %.2f'%(fut_pref_min, upper_bound_pref_fut))
        print('N. grid points for future weight: %d' % len(fut_pref_grid_opt))
        print('N past timesteps:', n_past_timesteps)
        fut_pref_grid = fut_pref_grid_opt

        for idx_fut_pref, fut_pref_val in enumerate(fut_pref_grid_opt):
            print('%d/%d' % (idx_fut_pref+1, len(fut_pref_grid_opt)))

            # Reset iterator
            iter_data_past = iter(data_loader_past)

            # Copy model and optimizer
            student_model_copy = copy.deepcopy(student_model)
            optimizer_copy = torch.optim.SGD(student_model_copy.parameters(),
                                             lr=eta, momentum=momentum,
                                             weight_decay=w_regularizer)
            optimizer_copy.load_state_dict(optimizer.state_dict())

            # Run dynamics using past data <- improvable: running multiple dynamics
            av_running_cost_ss = 0
            for t in range(n_past_timesteps):

                # Reset iterator
                if t>=len_batch:
                    iter_data_past = iter(data_loader_past)

                # Batch
                minibatch = next(iter_data_past)
                x, y = minibatch
                x, y = x.to(device), y.to(device)
                y_target = -y.detach().clone().to(device)
                minibatch = (x, y, y_target)

                # Action
                a = a_greedy(student_model_copy, optimizer_copy, minibatch, data_buffer, control_cost_weight,
                             future_weight=fut_pref_val*weight_future, lr=eta,
                             a_min=a_min, a_max=a_max, n_gridpoints=n_gridpoints)

                # Model predictions on current batch
                y_pred = student_model_copy(x)

                # Save costs
                with torch.no_grad():
                    nef_cost = 0.5 * criterion(y_pred.reshape(-1,1), y_target.reshape(-1,1))
                    per_cost = 0.5 * control_cost_weight * a**2
                    running_cost = nef_cost + per_cost
                    if t>=n_past_timesteps-window_steadystate:
                        av_running_cost_ss += running_cost

                # Update model
                y_perturbed = y*(1-a) + y_target*a
                loss = 0.5 * criterion(y_pred.reshape(-1,1), y_perturbed.reshape(-1,1))
                optimizer_copy.zero_grad()
                loss.backward()
                optimizer_copy.step()

            # Running cost for fut_pref_val
            av_running_cost_ss = av_running_cost_ss/window_steadystate
            running_cost_vs_fut_pref[idx_fut_pref] = av_running_cost_ss

        # Best fut_pref_val
        idx_fut_pref = np.argmin(running_cost_vs_fut_pref)
        fut_pref = fut_pref_grid_opt[idx_fut_pref]
        print('fut_pref found:', fut_pref)

    #******************************#
    #         Run dynamics         #
    #******************************#

    # Initialize arrays dynamics
    w_stud_dynamics = np.zeros((n_timesteps, dim_input))
    a_dynamics = np.zeros(n_timesteps)
    nef_cost_dynamics = np.zeros(n_timesteps)
    per_cost_dynamics = np.zeros(n_timesteps)
    cum_cost_dynamics = np.zeros(n_timesteps)
    d_dynamics = np.zeros(n_timesteps)
    accuracy_dynamics = np.zeros(n_timesteps)

    # Reset iterator
    iter_data_past = iter(data_loader_past)

    interval = int(n_timesteps/10)
    print('Poisoned training')
    for t in range(n_timesteps):

        # Reset iterator
        if t>=len_batch:
            iter_data_incoming = iter(data_loader_incoming)

        # Batch
        minibatch = next(iter_data_incoming)
        x, y = minibatch
        x, y = x.to(device), y.to(device)
        y_target = -y.detach().clone().to(device)
        minibatch = (x, y, y_target)

        # Action
        a = a_greedy(student_model, optimizer, minibatch, data_buffer, control_cost_weight,
                     future_weight=fut_pref*weight_future, lr=eta,
                     a_min=a_min, a_max=a_max, n_gridpoints=n_gridpoints)

        # Save current student vector and action
        w_stud_dynamics[t,:] = student_model.linear1.weight.detach().cpu().numpy()
        a_dynamics[t] = a

        # Model predictions on current batch
        y_pred = student_model(x)

        # Save costs, test distance and accuracy
        with torch.no_grad():
            # costs
            nef_cost_dynamics[t] = 0.5 * criterion(y_pred.reshape(-1,1), y_target.reshape(-1,1))
            per_cost_dynamics[t] = 0.5 * control_cost_weight * a**2
            running_cost = nef_cost_dynamics[t] + per_cost_dynamics[t]
            if t==0:
                cum_cost_dynamics[t] = running_cost
            else:
                cum_cost_dynamics[t] = cum_cost_dynamics[t-1] + running_cost * np.exp(-beta*eta*t)

            # Relative distance and accuracy
            y_pred_test = student_model(x_test)
            loss_test = 0.5 * criterion(y_pred_test.reshape(-1,1), y_test.reshape(-1,1))
            d = loss_test.item() / loss_target_test.item()
            d_dynamics[t] = d**0.5
            class_pred_test = student_model.predict(x_test)
            accuracy_dynamics[t] = torch.sum(class_pred_test.reshape(-1,)==labels_test.reshape(-1,))/len(y_test)

        # Update model
        y_perturbed = y*(1-a) + y_target*a
        loss = 0.5 * criterion(y_pred.reshape(-1,1), y_perturbed.reshape(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progression
        if t==0:
            print('%d/%d'%(t+1, n_timesteps))
        elif (t+1)%interval==0:
            print('%d/%d'%(t+1, n_timesteps))

        # Export results
        if export_while_running:
            if t%500==0:
                results['w_dynamics'] = w_stud_dynamics
                results['a_dynamics'] = a_dynamics
                results['nef_cost_dynamics'] = nef_cost_dynamics
                results['per_cost_dynamics'] = per_cost_dynamics
                results['cum_cost_dynamics'] = cum_cost_dynamics
                results['d_dynamics'] = d_dynamics
                results['running_cost_vs_fut_pref_opt_grid'] = running_cost_vs_fut_pref
                results['fut_pref_opt_grid'] = fut_pref_grid
                results['fut_pref'] = fut_pref
                results['accuracy_dynamics'] = accuracy_dynamics
                results['time_progress'] = np.arange(t)

                for name in results.keys():
                    data_to_export = results[name]
                    filename = name + '__@@@__' + exp_description
                    np.save(export_path + filename, data_to_export)

    print('Perturbed training completed!')

    # Results dictionary
    results['w_dynamics'] = w_stud_dynamics
    results['a_dynamics'] = a_dynamics
    results['nef_cost_dynamics'] = nef_cost_dynamics
    results['per_cost_dynamics'] = per_cost_dynamics
    results['cum_cost_dynamics'] = cum_cost_dynamics
    results['d_dynamics'] = d_dynamics
    results['running_cost_vs_fut_pref_opt_grid'] = running_cost_vs_fut_pref
    results['fut_pref_opt_grid'] = fut_pref_grid
    results['fut_pref'] = fut_pref
    results['accuracy_dynamics'] = accuracy_dynamics
    results['time_progress'] = np.arange(t)

    # Export results:
    if export_results:
        for name in results.keys():
            data_to_export = results[name]
            filename = name + '__@@@__' + exp_description
            np.save(export_path + filename, data_to_export)

    print('Done')

    return results
