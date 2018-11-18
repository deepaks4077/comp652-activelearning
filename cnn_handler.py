from __future__ import print_function
import tensorflow as tf
import random
import numpy as np
import sys

#tensorflow stuff to be replaced by pytorch
class tensorflow_mnist_basic():
	def __init__(self):
		self.output_layer_size = 10
		self.create_network()
		
	def create_network(self):
		######Here is the neural net model described in Tensor Flow MNIST example
		def weight_variable(shape):
			initial = tf.truncated_normal(shape, stddev=0.1)
			return tf.Variable(initial)

		def bias_variable(shape):
			initial = tf.constant(0.1, shape=shape)
		 	return tf.Variable(initial)

		def conv2d(x, W):
			return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		self.x = tf.placeholder(tf.float32, [None, 784], name='x')
		self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

		self.y_ = tf.placeholder(tf.float32, [None, self.output_layer_size],  name='y_')

		# Convolutional layer 1
		self.W_conv1 = weight_variable([5, 5, 1, 32])
		self.b_conv1 = bias_variable([32])

		self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1 = max_pool_2x2(self.h_conv1)

		# Convolutional layer 2
		self.W_conv2 = weight_variable([5, 5, 32, 64])
		self.b_conv2 = bias_variable([64])

		self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2 = max_pool_2x2(self.h_conv2)

		# Fully connected layer 1
		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])

		self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
		self.b_fc1 = bias_variable([1024])

		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

		# Dropout
		self.keep_prob  = tf.placeholder(tf.float32)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		# Fully connected layer 2 (Output layer)
		self.W_fc2 = weight_variable([1024, self.output_layer_size])
		self.b_fc2 = bias_variable([self.output_layer_size])

		self.y = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2, name='y')

		# Evaluation functions
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

		self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

		# Training algorithm
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

		return