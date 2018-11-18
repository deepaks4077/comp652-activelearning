from __future__ import print_function
import tensorflow as tf
import random
import numpy as np
import sys

#custom modules
import data_handler
import cnn_handler

class active_learner():
    def __init__(self, dataset_obj, cnn_obj):
        self.dataset_obj = dataset_obj
        self.cnn_obj = cnn_obj


    ###### Define Acquisition Functions here
    def random_values(self, num_values, sess = None):
        return [random.randint(0,9) for b in num_values]

    def max_min_softmax(self, input_data, sess):
        return [abs(np.max(s) - np.min(s)) for s in sess.run(self.cnn_obj.y, feed_dict={self.cnn_obj.x: input_data, self.cnn_obj.keep_prob: 1.0})]

    ######Ranks mnist training images according to a rank function (random for normal, 
    ######model evaluation for active learning)
    def choose_examples(self, sess, batch_size, rank_function, chosen = []):

        look_size = batch_size * 100
        
        training_lookup_index = range(0, len(self.dataset_obj.train_images))

        #do not look at examples that have already been seen
        raw_remain = list(set(training_lookup_index) - set(chosen))
        
        #shuffle the unseen examples to randomize order
        random.shuffle(raw_remain)
        
        looking_in = []
        
        #look at the first batch_size * 100 unseen exmaples (shuffled)
        if (look_size >= len(raw_remain)):
            looking_in = raw_remain
        else:
            looking_in = raw_remain[:look_size]
            
        remain_data = [self.dataset_obj.train_images[k] for k in looking_in]
        
        #rank the examples according to the rank function
        ranks = rank_function(remain_data, sess)
        scores = np.column_stack((looking_in, ranks))
        to_return = []

        scores = np.array(scores)
            
        selected = []
            
        #select examples based on their scores, enough to fill a batch
        if len(scores) >= batch_size:  
            sort = scores[np.argsort(scores[:,1])]
            selected = sort[:batch_size]
        else:
            selected = scores

        #return the index value for each chosen example
        return [int(s[0]) for s in selected]

    ######Big mess of a function that does a lot of things
    def run_batch(self, runs, size, max_steps, active, extra_sample, print_every):
        #collects information about each run
        batch_log = []

        #converts mnist test labels into new labels given label function
        #test_labels = self.dataset_obj.test_labels

        for i in range(runs):
            #previously trained on examples
            chosen = []

            #order that examples were selected to be trained on
            ordered = []

            #collects information about a given run
            run_log = []

            with tf.Session() as sess:
                #initializes the model
                sess.run(tf.initialize_all_variables())

                #rank function defaults to random, if active parameter passed
                #then will be the active learning rank function
                ranker = self.random_values
                
                if active:
                    ranker = self.max_min_softmax
                
                #iterate for each step (mini-batch)
                for step in range(max_steps):
                    #determines which train examples to look at in this mini-batch
                    next_batch = self.choose_examples(sess, size, ranker, chosen)

                    #if re-sampling turned on, will double size of the mini-batch
                    #by randomly sampling from previously trained on exmaples
                    to_train = next_batch
                    if extra_sample and len(chosen) > 0:
                        random.shuffle(chosen)
                        to_train = chosen[:size] + next_batch

                    chosen = chosen + next_batch
                    ordered = ordered + next_batch

                    #if all examples have been trained on, will start over (new epoch kind of)
                    if(len(chosen) == len(self.dataset_obj.train_labels)):
                        chosen = []

                    #grabs images and labels based on mini-batch
                    batch_xs = [self.dataset_obj.train_images[s] for s in to_train]
                    batch_ys = [self.dataset_obj.train_labels[s] for s in to_train]

                    #counts how many positive examples were in a mini-batch
                    positive_examples = 0
                    changed_ys = batch_ys
                    for ys in changed_ys:
                        positive_examples = positive_examples + ys[0]

                    #creates test results every print_every mini-batches
                    #passed in as a parameter
                    if (step % print_every) == 0:
                        rr = sess.run([self.cnn_obj.accuracy, self.cnn_obj.cross_entropy], feed_dict={self.cnn_obj.x: self.dataset_obj.test_images, self.cnn_obj.y_: self.dataset_obj.test_labels, self.cnn_obj.keep_prob: 1.0})
                        acc = rr[0]
                        ce = rr[1]
                        print(acc)
                        run_log.append([step, acc, ce, float(positive_examples)/len(to_train)])
                    
                    #trains the model using the mini-batch
                    sess.run(self.cnn_obj.train_step, feed_dict={self.cnn_obj.x: batch_xs, self.cnn_obj.y_: changed_ys, self.cnn_obj.keep_prob: 0.5})
                
                #after all the mini-batch training, run against test set and generate results
                final_rr = sess.run([self.cnn_obj.accuracy, self.cnn_obj.cross_entropy], feed_dict={self.cnn_obj.x: self.dataset_obj.test_images, self.cnn_obj.y_: self.dataset_obj.test_labels, self.cnn_obj.keep_prob: 1.0})
                final = final_rr[0]
                final_ce = final_rr[1]
                print(max_steps, final)
                run_log.append([max_steps, final, final_ce, float(positive_examples)/len(to_train)])
                batch_log.append(run_log)
                print("done with run ", i)

                #start multi-epoch portion (kind of pasted on at the end)
                epoch_logs = []

                #evaluate every this many labels
                label_range = 250

                #this many epochs
                epochs = 20

                #mini-batch size
                epoch_mini_batch_size = 50

                #creates labeled data sets at label_range increments and runs multi-epoch model on that 
                for label_size in range(len(ordered)/label_range):
                    labels_length = (label_size + 1) * label_range
                    result = self.epoch_sample(ordered[0: labels_length], epoch_mini_batch_size, epochs, sess)
                    epoch_logs.append([labels_length, epochs, result[0], result[1]])
                print("labels\tepoch\taccuracy\tcross entropy")
                for entry in epoch_logs:
                    print(entry[0], "\t", entry[1], "\t", entry[2], "\t", entry[3])
        print("donezo")
        return batch_log

    ######trains one model on a subset of mnist data for a certian number of epochs
    def epoch_sample(self, chosen, mini_batch_size, epochs, sess):
        #initializes the model (starts it over)
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            print("starting epoch ", epoch)

            #shuffle the labeled dataset, and do mini-batch training
            random.shuffle(chosen)
            batches = len(chosen) / mini_batch_size
            for i in range(batches):
                end = (i + 1) * mini_batch_size
                if(end > len(chosen)):
                    end = len(chosen) - 1
                training_data = [self.dataset_obj.train_images[s] for s in chosen[i * mini_batch_size: end]]
                training_labels = [self.dataset_obj.train_labels[s] for s in chosen[i * mini_batch_size: end]]
                sess.run(self.cnn_obj.train_step, feed_dict={self.cnn_obj.x: training_data, self.cnn_obj.y_: training_labels, self.cnn_obj.keep_prob: 0.5})
        print("done epoch training")
        epoch_rr = sess.run([self.cnn_obj.accuracy, self.cnn_obj.cross_entropy], feed_dict={self.cnn_obj.x: self.dataset_obj.test_images, self.cnn_obj.y_: self.dataset_obj.test_labels, self.cnn_obj.keep_prob: 1.0})
        epoch_acc = epoch_rr[0]
        epoch_ce = epoch_rr[1]
        print("labels: ", len(chosen))
        print("epochs: ", epochs)
        print("acc: ", epoch_acc)
        print("cross entropy: ", epoch_ce)
        return epoch_rr

    def print_average_series(self, to_average, columns, column_names):
        labels = [int(a[0]) for a in to_average[0]]
        column_collection = []
        for column in columns:
            transformed = []
            for run in to_average:
                transformed.append([a[column] for a in run])
            column_collection.append(np.mean(np.transpose(transformed), axis=1))
        column_collection.insert(0, np.array(labels))
        column_names.insert(0, 'iteration')
        print(*column_names, sep='\t')
        for row in np.transpose(column_collection):
            print(*row, sep='\t')

    def print_details(self, runs, batch_size, iterations, active, extra_sample):
        print("runs", runs)
        print("batch_size", batch_size)
        print("iterations", iterations)
        if(active):
            print("active")
        else:
            print("random")
        if(extra_sample):
            print("extra sample")
        else:
            print("standard sample")
        labels = self.dataset_obj.test_labels
        print("positive, negative", np.sum(labels, axis=0))

    ######not my most creative name.  does a run, and then prints out the results
    def make_a_good_test(self, runs, batch_size, iterations, active, extra_sample, print_every):
        self.print_details(runs, batch_size, iterations, active, extra_sample)
        results = self.run_batch(runs, batch_size, iterations, active, extra_sample, print_every)
        self.print_details(runs, batch_size, iterations, active, extra_sample)
        self.print_average_series(results, [1, 2, 3], ["accuracy", "cross entropy", "% positive examples"])



######runs a test with the following parameters:
###1st parameter: number of runs (almost always want just 1 or will be extremely long)
###2nd parameter: how bit each mini-batch should be
###3rd parameter: how many mini-batches in a run
###4th parameter: active learning turned on or not
###5th parameter: classification task ('old', '3s', 'split')
###6th parameter: re-samping turned on or not
###7th parameter: how often training is tested with test set
# example: python activemnist.py 1 10 2000 True old True 5
# would start one full run with mini-batches of size 10 for 2000 iterations with Active Learning
# turned on, classifying digits from 0-9, with re-sampling and printing test results every 5 mini-batches
# in addition it would then run 20 epochs at intervals of 250 label counts
#make_a_good_test(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == 'True', extra_sampling, print_every)


mnist_dataset = data_handler.mnist()
mnist_network = cnn_handler.tensorflow_mnist_basic()
al_obj = active_learner(mnist_dataset, mnist_network)


num_runs = 1
batch_size = 10
num_batch_runs = 500
active_learning = True
extra_sampling = True
print_every = 5

al_obj.make_a_good_test(num_runs, batch_size, num_batch_runs, active_learning, extra_sampling, print_every)