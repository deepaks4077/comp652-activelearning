Active Learning using Augmented Data from GANs/VAEs (Comp 652 Project)

Amit Sinha - 260819070
Anirudha Jitani - 260845002
Deepak Kumar - 260468268


Abstract

We investigate the effect of using different acquisition functions to figure out the benefits of using Active Learning for prioritizing labelling of unlabelled data. We use a dataset (CIFAR-10) which is completely labelled by an expert so that we can provide the expert labels when required by the Active Learning mechanism. We aim to show that fewer training examples are required when we consider relevant examples through the acquisition function to produce the same results on the test data. In other words, we benchmark in terms of labelled data required for the random based algorithm and the different acquisition functions to have the same performance.

Following this, we wish to observe the effect of using artificial data generated from pre-trained GANs/VAEs to augment the existing (real) training data set. Since a large amount of data can be generated using this method, we use the same Active Learning mechanism with the same acquisition functions to figure out the most relevant training examples for our task. Then we compare over the same test data (from the actual data set) and investigate the performance of the regular method versus the methods using artificially augmented data (among different acquisition functions). Primarily the goal here is to see if better performance can be achieved using artificially generated data.


Code Info

data_handler.py - Do all data related stuff here. Create a class and follow a similar structure for easy integration.

cnn_handler.py - Put a pytorch network in a class here, define all the optimization routines and loss functions in this class.

acquisition_functions.py - Add acquisition learning functions to the active_learner class. Will move some of the code into another script called 'experiments.py' which will import the active_learner class. Important stuff to be done here is to replace the tensorflow bits (a few lines) with the equivalent in pytorch.

pytorch_active_learner.py - Handles active learning stuff.

-----------------------------------------------------------------------------

To run the code properly, you must have torchvision. Also edit a file in the following:

$PATH_TO_YOUR_ANACONDA3/anaconda3/lib/python3.7/site-packages/torchvision-0.2.1-py3.7.egg/torchvision/datasets/mnist.py

Change Line 82 to:
return img, target, index