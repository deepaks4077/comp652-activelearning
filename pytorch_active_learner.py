import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np

#custom modules
from data_handler import mnist
from cnn_handler import convnet_mnist
import acquisition_functions


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

	def train(self):
		train_loader = self.dataset_manager.get_train_dataset_subset_loader([i for i in range(0,self.num_train_samples)])

		self.cnn_model.train()
		total_steps = len(train_loader)
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

			print('Test Accuracy on the {} test images: {} %'.format(total_test_samples, 100 * correct / total))

		return
 



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
		self.optimizer = torch.optim.Adam(cnn_model.parameters(), lr=self.learning_rate)

	def train_all(self):
		#train indices picked based on active learning scheme
		all_train_indices = [i for i in range(0, len(self.dataset_manager.train_dataset))]
		train_indices_to_use = []
		for num_train_samples in range(self.num_train_samples_per_step, self.max_num_train_samples + 1, self.num_train_samples_per_step):
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


		return

	def train(self, train_loader):
		self.cnn_model.train()
		total_steps = len(train_loader)
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
		self.test()
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

			print('Test Accuracy on the {} test images: {} %'.format(total_test_samples, 100 * correct / total))
		return


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#params for regular learner
num_epochs = 10
batch_size = 8
learning_rate = 0.001
num_train_samples = 1000
#Number of batch iterations during training between which test set is evaluated with the current model
test_every_n_iters = 25

dataset_manager = mnist(batch_size)
cnn_model = convnet_mnist(dataset_manager.num_classes).to(device)

rl = regular_learner(device, cnn_model, dataset_manager, num_epochs, batch_size, learning_rate, num_train_samples, test_every_n_iters)

rl.train()






#params for active learner
num_epochs = 5
batch_size = 8
learning_rate = 0.001
num_train_samples_per_step = 100
max_num_train_samples = 1000
#the number of samples you look at to rank amongst (if you have 10k samples you rank 1k random samples and pick the best N samples out of that)
#It should be greater than or equal to num_train_samples_per_step
num_samples_to_rank = 500
test_every_n_iters = 50

dataset_manager = mnist(batch_size)
cnn_model = convnet_mnist(dataset_manager.num_classes).to(device)

acquisition_func = acquisition_functions.max_min_softmax()

al = active_learner(device, cnn_model, dataset_manager, acquisition_func, num_epochs, batch_size, learning_rate, num_train_samples_per_step, max_num_train_samples, num_samples_to_rank, test_every_n_iters)

al.train_all()