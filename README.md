# OptimalControlAttacks

This repository contains the code implementing the attacks on data labels presented in the paper 'Attacks On Online Learners: a Teacher-Student Analysis' (https://arxiv.org/abs/2305.11132).

- **Real data greedy experiments**
  	
	MNIST-Perceptron:

	* Script: *_PerceptronMNIST - Input: index (int) for the C-grid.
	*	Imports parameters from ParametersPerceptronMNIST.py.
	*	Runs several experiments (n_runs_experiments) for the chosen parameters.
	*	Uses opt_pref=True, meaning that each simulation runs a calibration first.

	MNIST-LeNet:

	* Script: *_LeNetMNIST.py - Input: index (int) for the weight pre-factor grid.
	* 	Imports parameters from ParametersLeNetMNIST.py.
	* 	Runs several experiments (n_runs_experiments) for the chosen parameters.
	* 	Uses opt_pref=False - the experiment reaching the lowest steady-state.
	*	running average is used for downstream analysis.

	CIFAR10-VGG11 (transfer learning):

	* First train a teacher function using ScriptPytorchTeacherTraining_VGG11TransferCIFAR10.py.
	* Note that we use pre-trained weights imported from Torchvision and only train the last layer.
	* Then use ScriptPytorchExport_VGG11DataCIFAR10.py to export pre-last layer activations.
	* Finally, use the exported data to run experiments via ScriptPytorchExperiment_TransferVGG11CIFAR10.py.

	CIFAR10-ResNet18 (transfer learning):

	* Experiments follow the same protocol used for VGG11.
  

- **Synthetic data experiments**

  	The implementation of the experiments using synthetic data, including the clairvoyant and reinforcement learning attacks, will be uploaded soon. Currently available: greedy attacks on linear regression.
