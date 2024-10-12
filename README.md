# Attacks on Online Learners

This repository includes the code used to implement the experiments on data label attacks described in the paper *"Attacks On Online Learners: A Teacher-Student Analysis"* presented at [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46e37aeccafc3b4b697b17b8a36f3b30-Abstract-Conference.html).

- **Greedy attacks on real data**
  	
	Perceptron, MNIST:

	* Script: Experiment_PerceptronMNIST - Input: index (int) for the C-grid.
	*	Imports parameters from ParametersPerceptronMNIST.py.
	*	Runs several experiments (n_runs_experiments) for the chosen parameters.
	*	Uses opt_pref=True, meaning that each simulation first runs a calibration of the future weight pre-factor ($\tilde{\gamma}$).

	LeNet, MNIST:

	* Script: Experiment_LeNetMNIST.py - Input: index (int) for the future weight pre-factor ($\tilde{\gamma}$) grid.
	* 	Imports parameters from ParametersLeNetMNIST.py.
	* 	Runs several experiments (n_runs_experiments) for the chosen parameters.
	* 	Uses opt_pref=False - the experiment reaching the lowest steady-state running average is used for downstream analysis.

	VGG11, CIFAR10 (transfer learning):

	* First train a teacher function using TeacherTraining_VGG11TransferCIFAR10.py.
	* Note that we use pre-trained weights imported from Torchvision and only train the last layer.
	* Then use Export_VGG11DataCIFAR10.py to export pre-last layer activations.
	* Finally, use the exported data to run experiments via Experiment_TransferVGG11CIFAR10.py.

	ResNet18, CIFAR10 (transfer learning):

	* Experiments follow the same protocol used for VGG11.
  

- **Attacks on synthetic data**

  	Greedy attacks: 
	
	* Linear regression. Scripts: Experiment_GreedyAttacks_LinearRegression.py, Experiment_GreedyAttacks_LinearRegression_FractionPoisonedSamples.py, parameters loaded from ParametersGreedyAttacks_LinearRegression.py. Notebook: Experiment_GreedyAttacks_LinearRegression_MultiDimControl.ipynb.
	* Sigmoidal perceptron: Script Experiment_GreedyAttacks_SigmoidalPerceptron.py, parameters loaded from ParametersGreedyAttacks_SigmoidalPerceptron.py.
	* 2-layer NN. Script: Experiment_GreedyAttacks_2LayerNN.py, parameters loaded from ParametersGreedyAttacks_2LayerNN.py.
	
	Strategies comparison:
	
	* For each architecture (linear regression, sigmoidal perceptron, 2-layer NN):
	* 	Parameters are set in the corresponding file ParametersAttacksComparison_*.py.
	* 	Train the RL TD3 agent using the notebook RLAgentTraining_*.ipynb.
	*	Compare constant, reinforcement learning, greedy, and clairvoyant attacks using the notebook StrategiesComparison_*.ipynb.
