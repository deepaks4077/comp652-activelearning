#!/bin/bash

python pytorch_active_learner.py --dataset mnist --manualSeed 7 \
--rl_num_epochs 5 --rl_batch_size 10 --rl_learning_rate 0.001 --rl_train_samples_interval 250 --rl_max_train_samples 5000 --rl_test_every_n_iters 1000 \
--al_num_epochs 5 --al_batch_size 10 --al_learning_rate 0.001 --al_train_samples_interval 250 --al_max_train_samples 5000 --al_test_every_n_iters 1000 --al_num_samples_to_rank 2000