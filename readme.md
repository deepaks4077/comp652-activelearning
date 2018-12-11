# Active Learning using Augmented Data from GANs/VAEs

Deepak Sharma - 260468268
Amit Sinha - 260819070
Anirudha Jitani - 260845002

This work has been done for a project in the course COMP652/ECSE608

### Abstract

We explore the use of active learning in image classification task, where the im-
ages are drawn both from the real dataset and synthetic dataset. We evaluate differ-
ent acquisition functions by training a CNN that uses them on three different train-
ing sets - only real images, only synthetic images, both real and synthetic images.
We use MC dropout  to get the model prediction values for  multiple stochastic
runs, which we then use to score informative samples using the activation func-
tions. We observe that active learning can be helpful in achieving higher accuracy
with lower training samples as compared to traditional training methods in two
cases and provide an explanation as to why it doesn’t work in the case of synthetic
data.  In our next step,  we plan to incorporate active learning in the generation
process of the synthetic images, which could further enhance our classifier’s per-
formance.

# Relevant Code Info

The code is written in PyTorch (Version '0.4.1.post2') with Cuda Version 9.1

`main.py` - Drives the main operation of the code. Hypers are edited here. Sets up the comet.ml experiment for logging experimental metrics
`utils.py` - Handles the experiments and testing using Bayesian CNNs and controls acquisition functions
`models.py` - Define the architecture of the CNN
`AcquisitionFunctions.py` - Takes care of selecting the best N samples based on uncertainty information

# Experiments on Real Data Alone
Set lines 42,43 to:
```
fake_subset_indices = [x for x in range(0)]
real_subset_indices = [x for x in range(60000)]
```

# Experiments on Fake Data Alone
Set lines 42,43 to:
```
fake_subset_indices = [x for x in range(5000)]
real_subset_indices = [x for x in range(0)]
```

# Experiments on Real and Fake Data
Set lines 42,43 to:
```
fake_subset_indices = [x for x in range(3000)]
real_subset_indices = [x for x in range(30000)]
```

# Setting Relevant Hyper-parameters:
This can be done from lines 63-65:
```
NUM_TRIALS = 10
max_training_num = 5000
hyper_params = {"learning_rate": 0.001, "sampling_size": int(len(train_dataset)/6), "selection_size": 250, "max_training_num": max_training_num, "NUM_EPOCHS": 1, "bootstrap_samplesize": 20, "reset_model_per_selection": False}
```
`NUM_TRIALS` represents the number of samples for Monte Carlo Sampling.
`max_training_num` represents the maximum number of training samples used in the experiment.
`learning_rate` is the learning rate for the CNN used.
`sampling_size` is the pool on which we apply the acquisition functions. Ideally this value is the length of the entire. dataset, but to keep things more practical we use a subset of the entire data.
`selection_size` is the number of samples we choosed based on the Acquisition Function at each step.
`NUM_EPOCHS` refers to the number of epochs that are trained after each selection step.
`bootstrap_samplesize` refers to the initial sample size over which a a model is trained and initialized with for active learning.
`reset_model_per_selection` represents whether the model should be re-initialized after every selection step or not.


# Running the Experiment
The code is run using `main.py`:
```
python main.py
```

# Viewing the Results
View it on your comet.ml experiment by changing line 67:
```
experiment = Experiment(api_key="Gncqbz3Rhfy3MZJBcX7xKVJoo", project_name="comp652", workspace="comp652")
```
to your own experiment in comet.ml

OR

Don't change anything and view results here:
https://www.comet.ml/comp652/comp652
