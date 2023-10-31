##############################################################
#                                                            #
#                      Import libraries                      #
#                                                            #
##############################################################

import torchvision
from torchvision import datasets, transforms
import torchvision.models as models

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import copy
from collections import namedtuple
import numpy as np

import sys
import os
local_path = 'path_to_progect_folder/'
sys.path.append(local_path+'OptimalControlAttacks/')
from Modules import EmpiricalGreedyAttacksPytorch as EGAP



##############################################################
#                                                            #
#                        Parameters                          #
#                                                            #
##############################################################

# Architecture parameters
batchnorm_lastfc = True

# Dataset parameters
class1 = 3 # 3-cat, 4-deer
label_class1 = 1
class2 = 5 # 5-dog, 7-horse
label_class2 = -1
edge_length = 224 # 32 224
degrees = 10

# Dynamics parameters
batch_size = 10
n_epochs = 100

# Strings/paths
path_data = local_path + 'OptimalControlAttacks/ModelsData/CIFAR10/Classes_%d_%d/ResNetTransf/'%(class1, class2)
path_models = local_path + 'OptimalControlAttacks/Models/CIFAR10/Classes_%d_%d/ResNetTransf/'%(class1, class2)
description = 'classes#%d#%d_epochs#%d_batchnorm#%s'%(class1, class2, n_epochs, batchnorm_lastfc)

##############################################################
#                                                            #
#                   CIFAR10 preprocessing                    #
#                                                            #
##############################################################

## Trainset
trainset = datasets.CIFAR10('.data/',
            train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(edge_length),
                torchvision.transforms.RandomRotation(degrees),
                torchvision.transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2471, 0.2435, 0.2616))]))

# Mask classes
targets = np.array(trainset.targets)
mask = (targets==class1) | (targets==class2)
trainset.targets = list(targets[mask])
trainset.data = trainset.data[mask]

# Reassign class values
targets = targets[mask]
mask = targets==class1
targets[mask] = label_class1
mask = targets==class2
targets[mask] = label_class2
trainset.targets = list(targets)
trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

## Testset
testset = datasets.CIFAR10('.data/',
            train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(edge_length),
                torchvision.transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2471, 0.2435, 0.2616))]))

# Mask classes
targets = np.array(testset.targets)
mask = (targets==class1) | (targets==class2)
testset.targets = list(targets[mask])
testset.data = testset.data[mask]

# Reassign class values
targets = targets[mask]
mask = targets==class1
targets[mask] = label_class1
mask = targets==class2
targets[mask] = label_class2
testset.targets = list(targets)
testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)



##############################################################
#                                                            #
#                       Import model                         #
#                                                            #
##############################################################

# Teacher model
ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
resnet18_config = ResNetConfig(block = EGAP.BasicBlock, n_blocks = [2,2,2,2], channels = [64, 128, 256, 512])
teacher_model = EGAP.ResNet(resnet18_config, dobatchnorm=batchnorm_lastfc)
model_name = 'epoch%d_' % n_epochs + description + '.pth'
teacher_model.load_state_dict(torch.load(path_models+model_name, map_location=torch.device('cpu')))

# Extract teacher weights
parameters = teacher_model.lastfc.weight
dim_input = len(parameters.detach().flatten())
parameters_perceptron = (parameters.detach().cpu().flatten().numpy())*dim_input**0.5



##############################################################
#                                                            #
#                    Create training data                    #
#                                                            #
##############################################################

n_epochs = 10
n_samples = len(trainset_loader.dataset)
data_prelastfc_train = np.zeros((n_epochs*n_samples, dim_input))
data_label_train = np.zeros(n_epochs*n_samples)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher_model.to(device)
teacher_model.eval()
for epoch in range(n_epochs):
    for idx, (x, label) in enumerate(trainset_loader):
        x, label = x.to(device), label.to(device)
        prelastfc_batch = teacher_model.forward_prelastfc(x)
        idx_start = idx*batch_size + epoch*n_samples
        idx_end = (idx+1)*batch_size + epoch*n_samples
        data_prelastfc_train[idx_start:idx_end,:] = prelastfc_batch.detach().cpu().numpy()
        data_label_train[idx_start:idx_end] = label.detach().cpu().numpy()



##############################################################
#                                                            #
#                      Create test data                      #
#                                                            #
##############################################################

n_samples = len(testset_loader.dataset)
data_prelastfc_test = np.zeros((n_samples, dim_input))
data_label_test = np.zeros(n_samples)

for idx, (x, label) in enumerate(testset_loader):
    x, label = x.to(device), label.to(device)
    prelastfc_batch = teacher_model.forward_prelastfc(x)
    data_prelastfc_test[idx*batch_size:(idx+1)*batch_size,:] = prelastfc_batch.detach().cpu().numpy()
    data_label_test[idx*batch_size:(idx+1)*batch_size] = label.detach().cpu().numpy()



##############################################################
#                                                            #
#                        Export data                         #
#                                                            #
##############################################################

filename = 'train_prelastfc_input'
np.save(path_data+filename, data_prelastfc_train)
filename = 'train_prelastfc_label'
np.save(path_data+filename, data_label_train)

filename = 'test_prelastfc_input'
np.save(path_data+filename, data_prelastfc_test)
filename = 'test_prelastfc_label'
np.save(path_data+filename, data_label_test)

filename = 'teacher_parameters'
np.save(path_data+filename, parameters_perceptron)
