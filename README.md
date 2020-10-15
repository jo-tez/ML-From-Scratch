# Machine Learning From Scratch

**Note**: the `pythonista` branch contains a number of changes to make this repository usable in the Pythonista 3 iOS app.

---- 

## About
Python implementations of some of the fundamental Machine Learning models and algorithms from scratch.

The purpose of this project is not to produce as optimized and computationally efficient algorithms as possible
but rather to present the inner workings of them in a transparent and accessible way.

## Table of Contents
- [Machine Learning From Scratch][1]
  * [About][2]
  * [Table of Contents][3]
  * [Installation][4]
  * [Examples][5]
	+ [Polynomial Regression][6]
	+ [Classification With CNN][7]
	+ [Density-Based Clustering][8]
	+ [Generating Handwritten Digits][9]
	+ [Deep Reinforcement Learning][10]
	+ [Image Reconstruction With RBM][11]
	+ [Evolutionary Evolved Neural Network][12]
	+ [Genetic Algorithm][13]
	+ [Association Analysis][14]
  * [Implementations][15]
	+ [Supervised Learning][16]
	+ [Unsupervised Learning][17]
	+ [Reinforcement Learning][18]
	+ [Deep Learning][19]
  * [Contact][20]

## Installation
	$ git clone https://github.com/eriklindernoren/ML-From-Scratch
	$ cd ML-From-Scratch
	$ python setup.py install

## Examples
### Polynomial Regression
	$ python mlfromscratch/examples/polynomial_regression.py

<p align="center">
    <img src="http://eriklindernoren.se/images/p_reg.gif" width="640"\>
</p>
<p align="center">
    Figure: Training progress of a regularized polynomial regression model fitting <br>
    temperature data measured in Linköping, Sweden 2016.
</p>

### Classification With CNN
	$ python mlfromscratch/examples/convolutional_neural_network.py
	
	+---------+
	| ConvNet |
	+---------+
	Input Shape: (1, 8, 8)
	+----------------------+------------+--------------+
	| Layer Type           | Parameters | Output Shape |
	+----------------------+------------+--------------+
	| Conv2D               | 160        | (16, 8, 8)   |
	| Activation (ReLU)    | 0          | (16, 8, 8)   |
	| Dropout              | 0          | (16, 8, 8)   |
	| BatchNormalization   | 2048       | (16, 8, 8)   |
	| Conv2D               | 4640       | (32, 8, 8)   |
	| Activation (ReLU)    | 0          | (32, 8, 8)   |
	| Dropout              | 0          | (32, 8, 8)   |
	| BatchNormalization   | 4096       | (32, 8, 8)   |
	| Flatten              | 0          | (2048,)      |
	| Dense                | 524544     | (256,)       |
	| Activation (ReLU)    | 0          | (256,)       |
	| Dropout              | 0          | (256,)       |
	| BatchNormalization   | 512        | (256,)       |
	| Dense                | 2570       | (10,)        |
	| Activation (Softmax) | 0          | (10,)        |
	+----------------------+------------+--------------+
	Total Parameters: 538570
	
	Training: 100% [------------------------------------------------------------------------] Time: 0:01:55
	Accuracy: 0.987465181058

<p align="center">
    <img src="http://eriklindernoren.se/images/mlfs_cnn1.png" width="640">
</p>
<p align="center">
    Figure: Classification of the digit dataset using CNN.
</p>

### Density-Based Clustering
	$ python mlfromscratch/examples/dbscan.py

<p align="center">
    <img src="http://eriklindernoren.se/images/mlfs_dbscan.png" width="640">
</p>
<p align="center">
    Figure: Clustering of the moons dataset using DBSCAN.
</p>

### Generating Handwritten Digits
	$ python mlfromscratch/unsupervised_learning/generative_adversarial_network.py
	
	+-----------+
	| Generator |
	+-----------+
	Input Shape: (100,)
	+------------------------+------------+--------------+
	| Layer Type             | Parameters | Output Shape |
	+------------------------+------------+--------------+
	| Dense                  | 25856      | (256,)       |
	| Activation (LeakyReLU) | 0          | (256,)       |
	| BatchNormalization     | 512        | (256,)       |
	| Dense                  | 131584     | (512,)       |
	| Activation (LeakyReLU) | 0          | (512,)       |
	| BatchNormalization     | 1024       | (512,)       |
	| Dense                  | 525312     | (1024,)      |
	| Activation (LeakyReLU) | 0          | (1024,)      |
	| BatchNormalization     | 2048       | (1024,)      |
	| Dense                  | 803600     | (784,)       |
	| Activation (TanH)      | 0          | (784,)       |
	+------------------------+------------+--------------+
	Total Parameters: 1489936
	
	+---------------+
	| Discriminator |
	+---------------+
	Input Shape: (784,)
	+------------------------+------------+--------------+
	| Layer Type             | Parameters | Output Shape |
	+------------------------+------------+--------------+
	| Dense                  | 401920     | (512,)       |
	| Activation (LeakyReLU) | 0          | (512,)       |
	| Dropout                | 0          | (512,)       |
	| Dense                  | 131328     | (256,)       |
	| Activation (LeakyReLU) | 0          | (256,)       |
	| Dropout                | 0          | (256,)       |
	| Dense                  | 514        | (2,)         |
	| Activation (Softmax)   | 0          | (2,)         |
	+------------------------+------------+--------------+
	Total Parameters: 533762


<p align="center">
    <img src="http://eriklindernoren.se/images/gan_mnist5.gif" width="640">
</p>
<p align="center">
    Figure: Training progress of a Generative Adversarial Network generating <br>
    handwritten digits.
</p>

### Deep Reinforcement Learning
	$ python mlfromscratch/examples/deep_q_network.py
	
	+----------------+
	| Deep Q-Network |
	+----------------+
	Input Shape: (4,)
	+-------------------+------------+--------------+
	| Layer Type        | Parameters | Output Shape |
	+-------------------+------------+--------------+
	| Dense             | 320        | (64,)        |
	| Activation (ReLU) | 0          | (64,)        |
	| Dense             | 130        | (2,)         |
	+-------------------+------------+--------------+
	Total Parameters: 450

<p align="center">
    <img src="http://eriklindernoren.se/images/mlfs_dql1.gif" width="640">
</p>
<p align="center">
    Figure: Deep Q-Network solution to the CartPole-v1 environment in OpenAI gym.
</p>

### Image Reconstruction With RBM
	$ python mlfromscratch/examples/restricted_boltzmann_machine.py

<p align="center">
    <img src="http://eriklindernoren.se/images/rbm_digits1.gif" width="640">
</p>
<p align="center">
    Figure: Shows how the network gets better during training at reconstructing <br>
    the digit 2 in the MNIST dataset.
</p>

### Evolutionary Evolved Neural Network
	$ python mlfromscratch/examples/neuroevolution.py
	
	+---------------+
	| Model Summary |
	+---------------+
	Input Shape: (64,)
	+----------------------+------------+--------------+
	| Layer Type           | Parameters | Output Shape |
	+----------------------+------------+--------------+
	| Dense                | 1040       | (16,)        |
	| Activation (ReLU)    | 0          | (16,)        |
	| Dense                | 170        | (10,)        |
	| Activation (Softmax) | 0          | (10,)        |
	+----------------------+------------+--------------+
	Total Parameters: 1210
	
	Population Size: 100
	Generations: 3000
	Mutation Rate: 0.01
	
	[0 Best Individual - Fitness: 3.08301, Accuracy: 10.5%]
	[1 Best Individual - Fitness: 3.08746, Accuracy: 12.0%]
	...
	[2999 Best Individual - Fitness: 94.08513, Accuracy: 98.5%]
	Test set accuracy: 96.7%

<p align="center">
    <img src="http://eriklindernoren.se/images/evo_nn4.png" width="640">
</p>
<p align="center">
    Figure: Classification of the digit dataset by a neural network which has<br>
    been evolutionary evolved.
</p>

### Genetic Algorithm
	$ python mlfromscratch/examples/genetic_algorithm.py
	
	+--------+
	|   GA   |
	+--------+
	Description: Implementation of a Genetic Algorithm which aims to produce
	the user specified target string. This implementation calculates each
	candidate's fitness based on the alphabetical distance between the candidate
	and the target. A candidate is selected as a parent with probabilities proportional
	to the candidate's fitness. Reproduction is implemented as a single-point
	crossover between pairs of parents. Mutation is done by randomly assigning
	new characters with uniform probability.
	
	Parameters
	----------
	Target String: 'Genetic Algorithm'
	Population Size: 100
	Mutation Rate: 0.05
	
	[0 Closest Candidate: 'CJqlJguPlqzvpoJmb', Fitness: 0.00]
	[1 Closest Candidate: 'MCxZxdr nlfiwwGEk', Fitness: 0.01]
	[2 Closest Candidate: 'MCxZxdm nlfiwwGcx', Fitness: 0.01]
	[3 Closest Candidate: 'SmdsAklMHn kBIwKn', Fitness: 0.01]
	[4 Closest Candidate: '  lotneaJOasWfu Z', Fitness: 0.01]
	...
	[292 Closest Candidate: 'GeneticaAlgorithm', Fitness: 1.00]
	[293 Closest Candidate: 'GeneticaAlgorithm', Fitness: 1.00]
	[294 Answer: 'Genetic Algorithm']

### Association Analysis
	$ python mlfromscratch/examples/apriori.py
	+-------------+
	|   Apriori   |
	+-------------+
	Minimum Support: 0.25
	Minimum Confidence: 0.8
	Transactions:
	    [1, 2, 3, 4]
	    [1, 2, 4]
	    [1, 2]
	    [2, 3, 4]
	    [2, 3]
	    [3, 4]
	    [2, 4]
	Frequent Itemsets:
	    [1, 2, 3, 4, [1, 2], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 4], [2, 3, 4]]
	Rules:
	    1 -> 2 (support: 0.43, confidence: 1.0)
	    4 -> 2 (support: 0.57, confidence: 0.8)
	    [1, 4] -> 2 (support: 0.29, confidence: 1.0)


## Implementations
### Supervised Learning
- [Adaboost][21]
- [Bayesian Regression][22]
- [Decision Tree][23]
- [Elastic Net][24]
- [Gradient Boosting][25]
- [K Nearest Neighbors][26]
- [Lasso Regression][27]
- [Linear Discriminant Analysis][28]
- [Linear Regression][29]
- [Logistic Regression][30]
- [Multi-class Linear Discriminant Analysis][31]
- [Multilayer Perceptron][32]
- [Naive Bayes][33]
- [Neuroevolution][34]
- [Particle Swarm Optimization of Neural Network][35]
- [Perceptron][36]
- [Polynomial Regression][37]
- [Random Forest][38]
- [Ridge Regression][39]
- [Support Vector Machine][40]
- [XGBoost][41]

### Unsupervised Learning
- [Apriori][42]
- [Autoencoder][43]
- [DBSCAN][44]
- [FP-Growth][45]
- [Gaussian Mixture Model][46]
- [Generative Adversarial Network][47]
- [Genetic Algorithm][48]
- [K-Means][49]
- [Partitioning Around Medoids][50]
- [Principal Component Analysis][51]
- [Restricted Boltzmann Machine][52]

### Reinforcement Learning
- [Deep Q-Network][53]

### Deep Learning
  + [Neural Network][54]
  + [Layers][55]
	* Activation Layer
	* Average Pooling Layer
	* Batch Normalization Layer
	* Constant Padding Layer
	* Convolutional Layer
	* Dropout Layer
	* Flatten Layer
	* Fully-Connected (Dense) Layer
	* Fully-Connected RNN Layer
	* Max Pooling Layer
	* Reshape Layer
	* Up Sampling Layer
	* Zero Padding Layer
  + Model Types
	* [Convolutional Neural Network][56]
	* [Multilayer Perceptron][57]
	* [Recurrent Neural Network][58]

## Contact
If there's some implementation you would like to see here or if you're just feeling social,
feel free to [email][59] me or connect with me on [LinkedIn][60].

[1]:	#machine-learning-from-scratch
[2]:	#about
[3]:	#table-of-contents
[4]:	#installation
[5]:	#examples
[6]:	#polynomial-regression
[7]:	#classification-with-cnn
[8]:	#density-based-clustering
[9]:	#generating-handwritten-digits
[10]:	#deep-reinforcement-learning
[11]:	#image-reconstruction-with-rbm
[12]:	#evolutionary-evolved-neural-network
[13]:	#genetic-algorithm
[14]:	#association-analysis
[15]:	#implementations
[16]:	#supervised-learning
[17]:	#unsupervised-learning
[18]:	#reinforcement-learning
[19]:	#deep-learning
[20]:	#contact
[21]:	mlfromscratch/supervised_learning/adaboost.py
[22]:	mlfromscratch/supervised_learning/bayesian_regression.py
[23]:	mlfromscratch/supervised_learning/decision_tree.py
[24]:	mlfromscratch/supervised_learning/regression.py
[25]:	mlfromscratch/supervised_learning/gradient_boosting.py
[26]:	mlfromscratch/supervised_learning/k_nearest_neighbors.py
[27]:	mlfromscratch/supervised_learning/regression.py
[28]:	mlfromscratch/supervised_learning/linear_discriminant_analysis.py
[29]:	mlfromscratch/supervised_learning/regression.py
[30]:	mlfromscratch/supervised_learning/logistic_regression.py
[31]:	mlfromscratch/supervised_learning/multi_class_lda.py
[32]:	mlfromscratch/supervised_learning/multilayer_perceptron.py
[33]:	mlfromscratch/supervised_learning/naive_bayes.py
[34]:	mlfromscratch/supervised_learning/neuroevolution.py
[35]:	mlfromscratch/supervised_learning/particle_swarm_optimization.py
[36]:	mlfromscratch/supervised_learning/perceptron.py
[37]:	mlfromscratch/supervised_learning/regression.py
[38]:	mlfromscratch/supervised_learning/random_forest.py
[39]:	mlfromscratch/supervised_learning/regression.py
[40]:	mlfromscratch/supervised_learning/support_vector_machine.py
[41]:	mlfromscratch/supervised_learning/xgboost.py
[42]:	mlfromscratch/unsupervised_learning/apriori.py
[43]:	mlfromscratch/unsupervised_learning/autoencoder.py
[44]:	mlfromscratch/unsupervised_learning/dbscan.py
[45]:	mlfromscratch/unsupervised_learning/fp_growth.py
[46]:	mlfromscratch/unsupervised_learning/gaussian_mixture_model.py
[47]:	mlfromscratch/unsupervised_learning/generative_adversarial_network.py
[48]:	mlfromscratch/unsupervised_learning/genetic_algorithm.py
[49]:	mlfromscratch/unsupervised_learning/k_means.py
[50]:	mlfromscratch/unsupervised_learning/partitioning_around_medoids.py
[51]:	mlfromscratch/unsupervised_learning/principal_component_analysis.py
[52]:	mlfromscratch/unsupervised_learning/restricted_boltzmann_machine.py
[53]:	mlfromscratch/reinforcement_learning/deep_q_network.py
[54]:	mlfromscratch/deep_learning/neural_network.py
[55]:	mlfromscratch/deep_learning/layers.py
[56]:	mlfromscratch/examples/convolutional_neural_network.py
[57]:	mlfromscratch/examples/multilayer_perceptron.py
[58]:	mlfromscratch/examples/recurrent_neural_network.py
[59]:	mailto:eriklindernoren@gmail.com
[60]:	https://www.linkedin.com/in/eriklindernoren/