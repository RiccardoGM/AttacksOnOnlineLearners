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
edge_length = 224
degrees = 10

# Dynamics parameters
learning_rate = 1e-3
n_epochs = 50
saving_Depochs = 5
batch_size = 20

# Strings/paths
path_models = local_path + 'OptimalControlAttacks/Models/CIFAR10/Classes_%d_%d/VGGNetTransf/' % (class1, class2)
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
testset_loader = DataLoader(testset, batch_size=2000, shuffle=False)
testset_iterator = iter(testset_loader)
sample = next(testset_iterator)
test_input, test_label = sample[0], sample[1]
test_input_subset = test_input[0:500]
test_label_subset = test_label[0:500]



##############################################################
#                                                            #
#                       Define model                         #
#                                                            #
##############################################################

# Teacher model
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg11_layers = EGAP.get_vgg_layers(vgg11_config, batch_norm=True)
teacher_model = EGAP.VGG(vgg11_layers, denselayers_width=4096, dobatchnorm=batchnorm_lastfc)
teacher_model_dict = teacher_model.state_dict()

# Pre-trained weights
pretrained_model = models.vgg11_bn(pretrained=True)
pretrained_dict = pretrained_model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in teacher_model_dict and k not in ['classifier.6.weight', 'classifier.6.bias']}
# 2. overwrite entries in the existing state dict
teacher_model_dict.update(pretrained_dict)
# 3. load the new state dict
teacher_model.load_state_dict(teacher_model_dict)

# Freeze pre-trained layers
req_grad = False
for name, param in teacher_model.named_parameters():
    if name in pretrained_dict.keys():
        param.requires_grad = req_grad



##############################################################
#                                                            #
#                       Run training                         #
#                                                            #
##############################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher_model.to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
test_input_subset, test_label_subset = test_input_subset.to(device), test_label_subset.to(device)
loss_list = []
acc_list = []

# Run training
for epoch in range(n_epochs):

    # Backprop
    teacher_model.train()
    for idx, (batch_input, batch_label) in enumerate(trainset_loader):
        batch_input, batch_label = batch_input.to(device), batch_label.to(device)
        pred_label = teacher_model(batch_input)
        loss = criterion(pred_label.flatten().double(), batch_label.flatten().double())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate accuracy
    teacher_model.eval()
    with torch.no_grad():
        acc_num = torch.sum(torch.sign(teacher_model(test_input_subset)).flatten()==test_label_subset)
        acc_denom = len(test_label_subset)
        acc = acc_num/acc_denom

    loss_list.append(loss.detach().cpu().numpy())
    acc_list.append(acc.detach().cpu().numpy())

    # Save model
    if (epoch+1)%saving_Depochs==0:
        model_name = 'epoch%d_' % (epoch+1)
        model_name = model_name + description + '.pth'
        torch.save(teacher_model.state_dict(), path_models+model_name)

    # Print progress
    print('epoch: {}, loss: {}, acc: {}'.format(epoch, loss, acc))

print('Training complete!')

# Save training dynamics
loss_arr = np.array(loss_list)
name_file = 'lossdynamics_' + description
np.save(path_models+name_file, loss_arr)
acc_arr = np.array(acc_list)
name_file = 'accdynamics_' + description
np.save(path_models+name_file, acc_arr)
