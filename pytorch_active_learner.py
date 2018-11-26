import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import argparse
import random
import matplotlib.pyplot as plt

#custom modules
from data_handler import mnist, cifar10
from cnn_handler import convnet_mnist, convnet_cifar10
import acquisition_functions

THRESHOLD_FOR_FAKE_DATA_GENERATION = 50000

class regular_learner():
	def __init__(self, device, cnn_model, dataset_manager, num_epochs = 10, batch_size = 8, learning_rate = 0.001, num_train_samples = 500, test_every_n_iters = 100):
		self.device = device
		self.cnn_model = cnn_model
		self.dataset_manager = dataset_manager
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.num_train_samples = num_train_samples
		self.test_every_n_iters = test_every_n_iters
		
		# Loss and optimizer
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(cnn_model.parameters(), lr=self.learning_rate)

		#stuff to plot later
		self.test_accuracy_final = -1
		self.corresponding_num_train_labels = -1
		self.fake_data_fraction = -1


	def train(self):
		train_loader = self.dataset_manager.get_random_subset_train_dataset_loader(self.num_train_samples)

		self.cnn_model.train()
		total_steps = len(train_loader)
		num = 0; den = 0
		for epoch in range(self.num_epochs):
			for i, (images, labels, tl_ind) in enumerate(train_loader):
				images = images.to(self.device)
				labels = labels.to(self.device)
				
				# Forward pass
				outputs = self.cnn_model(images)
				loss = self.criterion(outputs, labels)
				
				# Backward and optimize
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				if (i+1) % self.test_every_n_iters == 0:
					print ('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
					self.test()
				if epoch == 0:
					for ind in tl_ind:
						if int(ind) >= THRESHOLD_FOR_FAKE_DATA_GENERATION:
							num = num + 1
						den = den + 1

		self.test_accuracy_final = self.test()
		self.corresponding_num_train_labels = self.num_train_samples
		self.fake_data_fraction = float(num) / float(den)
		return

	def test(self):
		self.cnn_model.eval()
		
		total_test_samples = len(self.dataset_manager.test_dataset)
		with torch.no_grad():
		
			correct = 0
			total = 0
			
			for images, labels, tl_ind in self.dataset_manager.test_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				outputs = self.cnn_model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

			test_accuracy = 100 * correct / total
			print('Test Accuracy on the {} test images: {} %'.format(total_test_samples, test_accuracy))

		return test_accuracy
 



class active_learner():
	def __init__(self, device, cnn_model, dataset_manager, acquisition_func, num_epochs = 10, batch_size = 8, learning_rate = 0.001, num_train_samples_per_step = 250, max_num_train_samples = 1000, num_samples_to_rank = 1000, test_every_n_iters = 100):
		self.device = device
		self.cnn_model = cnn_model
		self.dataset_manager = dataset_manager
		self.acquisition_func = acquisition_func
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.num_train_samples_per_step = num_train_samples_per_step
		self.max_num_train_samples = max_num_train_samples
		self.num_samples_to_rank = num_samples_to_rank
		self.test_every_n_iters = test_every_n_iters
		
		# Loss and optimizer
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate)

		#stuff to plot later
		self.test_accuracy_each_label_set = []
		self.corresponding_num_train_labels = []
		self.fake_data_fraction = []

	def train_all(self):
		#train indices picked based on active learning scheme
		all_train_indices = [i for i in range(0, len(self.dataset_manager.train_dataset))]
		train_indices_to_use = []

		self.cnn_model.reset()
		self.cnn_model = self.cnn_model.to(self.device)
		self.optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate)
		for num_train_samples in range(self.num_train_samples_per_step, self.max_num_train_samples + 1, self.num_train_samples_per_step):
			# self.cnn_model.reset()
			# self.cnn_model = self.cnn_model.to(self.device)
			# self.optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate)
			
			print ('----------------------------------------------------------------')
			print ('Active Learning: Now Training with {} Training Samples'.format(num_train_samples))
			#train indices not already used in active learning
			leftover_train_indices = list(set(all_train_indices) - set(train_indices_to_use))
			leftover_train_loader = self.dataset_manager.get_train_dataset_subset_loader(leftover_train_indices)
			best_train_indices_from_leftovers = self.acquisition_func.get_best_sample_indices(self.device, self.cnn_model, leftover_train_loader, self.num_train_samples_per_step, self.num_samples_to_rank)
			train_indices_to_use = train_indices_to_use + best_train_indices_from_leftovers

			#now train with the subset of the data that is best for active learning
			# print('train indices to use')
			# print (train_indices_to_use)
			train_loader_to_use = self.dataset_manager.get_train_dataset_subset_loader(train_indices_to_use)

			self.train(train_loader_to_use)

			self.corresponding_num_train_labels.append(num_train_samples)
		return

	def train(self, train_loader):
		self.cnn_model.train()
		total_steps = len(train_loader)
		num = 0; den = 0
		for epoch in range(self.num_epochs):
			for i, (images, labels, tl_ind) in enumerate(train_loader):
				images = images.to(self.device)
				labels = labels.to(self.device)
				
				# Forward pass
				outputs = self.cnn_model(images)
				loss = self.criterion(outputs, labels)
				
				# Backward and optimize
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				if (i+1) % self.test_every_n_iters == 0:
					print ('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
					self.test()

				if epoch == 0:
					for ind in tl_ind:
						if int(ind) >= THRESHOLD_FOR_FAKE_DATA_GENERATION:
							num = num + 1
						den = den + 1

		self.test_accuracy_each_label_set.append(self.test())
		self.fake_data_fraction.append(float(num) / float(den))
		return

	def test(self):
		self.cnn_model.eval()
		
		total_test_samples = len(self.dataset_manager.test_dataset)
		with torch.no_grad():
		
			correct = 0
			total = 0
			
			for images, labels, tl_ind in self.dataset_manager.test_loader:
				images = images.to(self.device)
				labels = labels.to(self.device)
				outputs = self.cnn_model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

			test_accuracy = 100 * correct / total
			print('Test Accuracy on the {} test images: {} %'.format(total_test_samples, test_accuracy))
		return test_accuracy


# # Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# #params for regular learner
# num_epochs = 5
# batch_size = 10
# learning_rate = 0.001
# num_train_samples = 20000
# #Number of batch iterations during training between which test set is evaluated with the current model
# test_every_n_iters = 1000

# #mnist
# dataset_manager = mnist(batch_size)
# cnn_model = convnet_mnist(dataset_manager.num_classes).to(device)
# #cifar10
# # dataset_manager = cifar10(batch_size)
# # cnn_model = convnet_cifar10(dataset_manager.num_classes).to(device)

# rl = regular_learner(device, cnn_model, dataset_manager, num_epochs, batch_size, learning_rate, num_train_samples, test_every_n_iters)

# rl.train()

# print (rl.corresponding_num_train_labels)
# print (rl.test_accuracy_final)





# #params for active learner
# num_epochs = 5
# batch_size = 10
# learning_rate = 0.001
# num_train_samples_per_step = 500
# max_num_train_samples = 20000
# #the number of samples you look at to rank amongst (if you have 10k samples you rank 1k random samples and pick the best N samples out of that)
# #It should be greater than or equal to num_train_samples_per_step
# num_samples_to_rank = 2000
# test_every_n_iters = 1000

# #mnist
# dataset_manager = mnist(batch_size)
# cnn_model = convnet_mnist(dataset_manager.num_classes).to(device)
# #cifar10
# # dataset_manager = cifar10(batch_size)
# # cnn_model = convnet_cifar10(dataset_manager.num_classes).to(device)

# #acquisition_func = acquisition_functions.max_min_softmax()
# #acquisition_func = acquisition_functions.smallest_margin_softmax()
# #acquisition_func = acquisition_functions.entropy_softmax()
# #acquisition_func = acquisition_functions.uncertainty_density_max_softmax()
# acquisition_func = acquisition_functions.uncertainty_density_entropy_softmax()

# al = active_learner(device, cnn_model, dataset_manager, acquisition_func, num_epochs, batch_size, learning_rate, num_train_samples_per_step, max_num_train_samples, num_samples_to_rank, test_every_n_iters)
# al.train_all()

# print (al.corresponding_num_train_labels)
# print (al.test_accuracy_each_label_set)





if __name__ == "__main__":
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True, help='mnist | cifar10')

	parser.add_argument('--rl_num_epochs', type=int, default = 10, help='Number of Epochs for regular learner')
	parser.add_argument('--rl_batch_size', type=int, default = 10, help='Batch Size for regular learner')
	parser.add_argument('--rl_learning_rate', type=float, default = 0.001, help='Learning Rate for regular learner')
	parser.add_argument('--rl_train_samples_interval', type=int, default = 500, help='Incremental Value for number of training samples for regular learner')
	parser.add_argument('--rl_max_train_samples', type=int, default = 20000, help='Max Value for number of training samples for regular learner')
	parser.add_argument('--rl_test_every_n_iters', type=int, default = 500, help='How frequently test error is evaluated for regular learner')

	parser.add_argument('--al_num_epochs', type=int, default = 10, help='Number of Epochs as integer for active learner')
	parser.add_argument('--al_batch_size', type=int, default = 10, help='Batch Size for active learner')
	parser.add_argument('--al_learning_rate', type=float, default = 0.001, help='Learning Rate for active learner')
	parser.add_argument('--al_train_samples_interval', type=int, default = 500, help='Incremental Value for number of training samples for active learner')
	parser.add_argument('--al_max_train_samples', type=int, default = 20000, help='Max Value for number of training samples for active learner')
	parser.add_argument('--al_num_samples_to_rank', type=int, default = 2000, help='The random pool to look at for ranking for active learner')
	parser.add_argument('--al_test_every_n_iters', type=int, default = 500, help='How frequently test error is evaluated for active learner')

	parser.add_argument('--manualSeed', type=int, help='Put in a fixed random seed for reproducibility')	

	opt = parser.parse_args()
	print(opt)

	if opt.manualSeed is None:
		opt.manualSeed = random.randint(1, 10000)
	

	#regular learner part
	#PLOT THESE
	rl_test_accuracies = []
	rl_num_train_lables = []
	rl_fake_data_fraction = []

	print("Random Seed: ", opt.manualSeed)
	random.seed(opt.manualSeed)
	torch.manual_seed(opt.manualSeed)
	
	batch_size = opt.rl_batch_size

	if opt.dataset == 'mnist':
		dataset_manager = mnist(batch_size)
		cnn_model = convnet_mnist(dataset_manager.num_classes).to(device)
	elif opt.dataset == 'cifar10':
		dataset_manager = cifar10(batch_size)
		cnn_model = convnet_cifar10(dataset_manager.num_classes).to(device)
	else:
		assert False, "Dataset must be cifar10 or mnist. Given `{}`".format(opt.dataset)

	num_epochs = opt.rl_num_epochs
	learning_rate = opt.rl_learning_rate
	train_samples_interval = opt.rl_train_samples_interval
	max_train_samples = opt.rl_max_train_samples
	test_every_n_iters = opt.rl_test_every_n_iters

	for num_train_samples in range(train_samples_interval, max_train_samples + 1, train_samples_interval):
		cnn_model.reset()
		cnn_model = cnn_model.to(device)
		print ('----------------------------------------------------------------')
		print ('Regular Learning: Now Training with {} Training Samples'.format(num_train_samples))
		rl = regular_learner(device, cnn_model, dataset_manager, num_epochs, batch_size, learning_rate, num_train_samples, test_every_n_iters)
		rl.train()

		rl_num_train_lables.append(rl.corresponding_num_train_labels)
		rl_test_accuracies.append(rl.test_accuracy_final)
		rl_fake_data_fraction.append(rl.fake_data_fraction)


	#active learner part

	print("Random Seed: ", opt.manualSeed)
	random.seed(opt.manualSeed)
	torch.manual_seed(opt.manualSeed)
	
	batch_size = opt.al_batch_size

	if opt.dataset == 'mnist':
		dataset_manager = mnist(batch_size)
		cnn_model = convnet_mnist(dataset_manager.num_classes).to(device)
	elif opt.dataset == 'cifar10':
		dataset_manager = cifar10(batch_size)
		cnn_model = convnet_cifar10(dataset_manager.num_classes).to(device)
	else:
		assert False, "Dataset must be cifar10 or mnist. Given `{}`".format(opt.dataset)

	num_epochs = opt.al_num_epochs
	learning_rate = opt.al_learning_rate
	train_samples_interval = opt.al_train_samples_interval
	max_train_samples = opt.al_max_train_samples
	num_samples_to_rank = opt.al_num_samples_to_rank
	test_every_n_iters = opt.al_test_every_n_iters

	af_max_min = acquisition_functions.max_min_softmax()
	af_small_margin = acquisition_functions.smallest_margin_softmax()
	af_entropy = acquisition_functions.entropy_softmax()
	af_density_max = acquisition_functions.uncertainty_density_max_softmax()
	af_density_entropy = acquisition_functions.uncertainty_density_entropy_softmax()

	al_max_min = active_learner(device, cnn_model, dataset_manager, af_max_min, num_epochs, batch_size, learning_rate, train_samples_interval, max_train_samples, num_samples_to_rank, test_every_n_iters)
	al_max_min.train_all()

	al_small_margin = active_learner(device, cnn_model, dataset_manager, af_small_margin, num_epochs, batch_size, learning_rate, train_samples_interval, max_train_samples, num_samples_to_rank, test_every_n_iters)
	al_small_margin.train_all()
	
	al_entropy = active_learner(device, cnn_model, dataset_manager, af_entropy, num_epochs, batch_size, learning_rate, train_samples_interval, max_train_samples, num_samples_to_rank, test_every_n_iters)
	al_entropy.train_all()

	al_density_max = active_learner(device, cnn_model, dataset_manager, af_density_max, num_epochs, batch_size, learning_rate, train_samples_interval, max_train_samples, num_samples_to_rank, test_every_n_iters)
	al_density_max.train_all()

	al_density_entropy = active_learner(device, cnn_model, dataset_manager, af_density_entropy, num_epochs, batch_size, learning_rate, train_samples_interval, max_train_samples, num_samples_to_rank, test_every_n_iters)
	al_density_entropy.train_all()


	#fake data fraction plots
	plt.figure('Fake Data Fraction Plot')
	plt.plot(rl_num_train_lables, rl_fake_data_fraction, color=[0.,0.,0.], label='Random')
	plt.plot(al_max_min.corresponding_num_train_labels, al_max_min.fake_data_fraction, color=[0.,1.,0.], label='Max-Min')
	plt.plot(al_small_margin.corresponding_num_train_labels, al_small_margin.fake_data_fraction, color=[0.9,0.1,0.5], label='Small Margin')
	plt.plot(al_entropy.corresponding_num_train_labels, al_entropy.fake_data_fraction, color=[0.1,0.7,0.3], label='Entropy')
	plt.plot(al_density_max.corresponding_num_train_labels, al_density_max.fake_data_fraction, color=[0.6,0.6,0.1], label='Dense-Max')
	plt.plot(al_density_entropy.corresponding_num_train_labels, al_density_entropy.fake_data_fraction, color=[0.1,0.6,0.9], label='Dense-Entropy')

	plt.xlabel('Number of Training Labels')
	plt.ylabel('Fake Data Fraction')
	plt.title('Number of Training Labels vs. Fake Data Fraction for Different Learning Approaches')
	plt.legend(loc='upper left')
	plt.show()

	#test error plots
	plt.figure('Test Error Plot')
	plt.plot(rl_num_train_lables, rl_test_accuracies, color=[0.,0.,0.], label='Random')

	#PLOT THESE
	# al_max_min.corresponding_num_train_labels
	# al_max_min.test_accuracy_each_label_set
	plt.plot(al_max_min.corresponding_num_train_labels, al_max_min.test_accuracy_each_label_set, color=[0.,1.,0.], label='Max-Min')

	# al_small_margin.corresponding_num_train_labels
	# al_small_margin.test_accuracy_each_label_set
	plt.plot(al_small_margin.corresponding_num_train_labels, al_small_margin.test_accuracy_each_label_set, color=[0.9,0.1,0.5], label='Small Margin')

	# al_entropy.corresponding_num_train_labels
	# al_entropy.test_accuracy_each_label_set
	plt.plot(al_entropy.corresponding_num_train_labels, al_entropy.test_accuracy_each_label_set, color=[0.1,0.7,0.3], label='Entropy')

	# al_density_max.corresponding_num_train_labels
	# al_density_max.test_accuracy_each_label_set
	plt.plot(al_density_max.corresponding_num_train_labels, al_density_max.test_accuracy_each_label_set, color=[0.6,0.6,0.1], label='Dense-Max')

	# al_density_entropy.corresponding_num_train_labels
	# al_density_entropy.test_accuracy_each_label_set
	plt.plot(al_density_entropy.corresponding_num_train_labels, al_density_entropy.test_accuracy_each_label_set, color=[0.1,0.6,0.9], label='Dense-Entropy')



	plt.xlabel('Number of Training Labels')
	plt.ylabel('Test Error')
	plt.title('Number of Training Labels vs. Test Errors for Different Learning Approaches')
	plt.legend(loc='upper left')
	plt.show()