from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cPickle

class mnist():
	def __init__(self):
		######Pulls reads in MNSIT data
		data = input_data.read_data_sets("MNIST_data/", one_hot=True)
		self.train_images = data.train.images
		self.test_images = data.test.images
		self.train_labels = data.train.labels
		self.test_labels = data.test.labels
		del(data)

	def preprocessing(self):
		#Put whatever normalization
		return

class cifar10():
	def __init__(self):
		######Edit cifar10 path here
		cifar_path = '/home/fishy/data/cifar10/cifar-10-batches-py/'
		self.train_images, self.test_images, self.train_labels, self.test_labels = self.load_full_data(cifar_path)

	def preprocessing(self):
		#Put whatever normalization
		self.train_images = self.train_images / 255.0
		self.test_images = self.test_images / 255.0
		return


	#stuff for reading cifar10 batches in python
	def load_full_data(self, path):
		def one_hot_encoding(decoded_input):
			ohe = np.zeros((decoded_input.size, decoded_input.max()+1))
			ohe[np.arange(decoded_input.size),decoded_input] = 1
			return ohe

		def load_batch(path, file):
			f = open(path+file, 'rb')
			dict = cPickle.load(f)
			images = dict['data']
			#images = np.reshape(images, (10000, 3, 32, 32))
			labels = dict['labels']
			imagearray = np.array(images)   #   (10000, 3072)
			labelarray = np.array(labels)   #   (10000,)
			return imagearray, labelarray

		imagearray, labelarray = load_batch(path, 'data_batch_1')
		train_images = imagearray
		train_labels = labelarray

		imagearray, labelarray = load_batch(path, 'data_batch_2')
		train_images = np.concatenate((train_images, imagearray), axis=0)
		train_labels = np.concatenate((train_labels, labelarray), axis=0)

		imagearray, labelarray = load_batch(path, 'data_batch_3')
		train_images = np.concatenate((train_images, imagearray), axis=0)
		train_labels = np.concatenate((train_labels, labelarray), axis=0)

		imagearray, labelarray = load_batch(path, 'data_batch_4')
		train_images = np.concatenate((train_images, imagearray), axis=0)
		train_labels = np.concatenate((train_labels, labelarray), axis=0)

		imagearray, labelarray = load_batch(path, 'data_batch_5')
		train_images = np.concatenate((train_images, imagearray), axis=0)
		train_labels = np.concatenate((train_labels, labelarray), axis=0)

		test_images, test_labels = load_batch(path, 'test_batch')

		train_labels = one_hot_encoding(train_labels)
		test_labels = one_hot_encoding(test_labels)

		return train_images, test_images, train_labels, test_labels

