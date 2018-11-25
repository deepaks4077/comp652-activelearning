import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np

#custom modules
from data_handler import mnist

class max_min_softmax():
	def __init__(self):
		return
	#And by best we mean the ones that give the lowest confidence in the network prediction
	def get_best_sample_indices(self, device, cnn_model, leftover_train_loader, num_train_samples_per_step, num_samples_to_rank):
		best_sample_indices = []
		
		softmax_fn = nn.Softmax()
		cnn_model.eval()
		
		with torch.no_grad():
			
			inds_chosen_to_rank = []
			confidence_chosen_to_rank = torch.tensor([])

			for i, (images, labels, tl_ind) in enumerate(leftover_train_loader):
				if (i*leftover_train_loader.batch_size >= num_samples_to_rank):
					break
				images = images.to(device)
				outputs = cnn_model(images)
				
				sm_outputs = softmax_fn(outputs)
				output_confidence = torch.max(sm_outputs, dim=1)[0] - torch.min(sm_outputs, dim=1)[0]
				output_confidence = output_confidence.cpu()

				confidence_chosen_to_rank = torch.cat((confidence_chosen_to_rank, output_confidence))
				inds_chosen_to_rank = inds_chosen_to_rank + tl_ind.tolist()

			
			if (len(inds_chosen_to_rank) <= num_train_samples_per_step):
				best_sample_indices = inds_chosen_to_rank
			else:
				#Now find the worst 'num_train_samples_per_step' and send the corresponding 'inds_chosen_to_rank' to best_sample_indices
				confidence_chosen_to_rank = np.array(confidence_chosen_to_rank.tolist())
				inds_chosen_to_rank = np.array(inds_chosen_to_rank)
				best_sample_indices = inds_chosen_to_rank[confidence_chosen_to_rank.argsort()[0:num_train_samples_per_step]].tolist()

		return best_sample_indices


class smallest_margin_softmax():
	def __init__(self):
		return
	#And by best we mean the ones that give the lowest confidence in the network prediction
	def get_best_sample_indices(self, device, cnn_model, leftover_train_loader, num_train_samples_per_step, num_samples_to_rank):
		best_sample_indices = []
		
		softmax_fn = nn.Softmax()
		cnn_model.eval()
		
		with torch.no_grad():
			
			inds_chosen_to_rank = []
			confidence_chosen_to_rank = torch.tensor([])

			for i, (images, labels, tl_ind) in enumerate(leftover_train_loader):
				if (i*leftover_train_loader.batch_size >= num_samples_to_rank):
					break
				images = images.to(device)
				outputs = cnn_model(images)
				
				sm_outputs = softmax_fn(outputs)
				sm_outputs_top_k = torch.topk(sm_outputs, 2)[0]
				output_confidence = sm_outputs_top_k[0] - sm_outputs_top_k[1]
				output_confidence = output_confidence.cpu()

				confidence_chosen_to_rank = torch.cat((confidence_chosen_to_rank, output_confidence))
				inds_chosen_to_rank = inds_chosen_to_rank + tl_ind.tolist()

			
			if (len(inds_chosen_to_rank) <= num_train_samples_per_step):
				best_sample_indices = inds_chosen_to_rank
			else:
				#Now find the worst 'num_train_samples_per_step' and send the corresponding 'inds_chosen_to_rank' to best_sample_indices
				confidence_chosen_to_rank = np.array(confidence_chosen_to_rank.tolist())
				inds_chosen_to_rank = np.array(inds_chosen_to_rank)
				best_sample_indices = inds_chosen_to_rank[confidence_chosen_to_rank.argsort()[0:num_train_samples_per_step]].tolist()

		return best_sample_indices


class entropy_softmax():
	def __init__(self):
		return
	#And by best we mean the ones that give the lowest confidence in the network prediction
	def get_best_sample_indices(self, device, cnn_model, leftover_train_loader, num_train_samples_per_step, num_samples_to_rank):
		best_sample_indices = []
		
		softmax_fn = nn.Softmax()
		cnn_model.eval()
		
		with torch.no_grad():
			
			inds_chosen_to_rank = []
			confidence_chosen_to_rank = torch.tensor([])

			for i, (images, labels, tl_ind) in enumerate(leftover_train_loader):
				if (i*leftover_train_loader.batch_size >= num_samples_to_rank):
					break
				images = images.to(device)
				outputs = cnn_model(images)
				
				sm_outputs = softmax_fn(outputs)
				#taking minus of entropy puts 
				output_confidence = torch.sum(sm_outputs * torch.log(sm_outputs), dim=1)
				output_confidence = output_confidence.cpu()

				confidence_chosen_to_rank = torch.cat((confidence_chosen_to_rank, output_confidence))
				inds_chosen_to_rank = inds_chosen_to_rank + tl_ind.tolist()
			
			if (len(inds_chosen_to_rank) <= num_train_samples_per_step):
				best_sample_indices = inds_chosen_to_rank
			else:
				#Now find the worst 'num_train_samples_per_step' and send the corresponding 'inds_chosen_to_rank' to best_sample_indices
				confidence_chosen_to_rank = np.array(confidence_chosen_to_rank.tolist())
				inds_chosen_to_rank = np.array(inds_chosen_to_rank)
				best_sample_indices = inds_chosen_to_rank[confidence_chosen_to_rank.argsort()[0:num_train_samples_per_step]].tolist()

		return best_sample_indices




class uncertainty_density_max_softmax():
	def __init__(self):
		return
	#And by best we mean the ones that give the lowest confidence in the network prediction
	def get_best_sample_indices(self, device, cnn_model, leftover_train_loader, num_train_samples_per_step, num_samples_to_rank):
		best_sample_indices = []
		
		softmax_fn = nn.Softmax()
		cnn_model.eval()
		
		with torch.no_grad():
			
			inds_chosen_to_rank = []
			output_uncertainty_all = torch.tensor([]).to(device)
			normalized_images_matrix = torch.tensor([]).to(device)

			for i, (images, labels, tl_ind) in enumerate(leftover_train_loader):
				if (i*leftover_train_loader.batch_size >= num_samples_to_rank):
					break
				images = images.to(device)
				normalized_images_matrix = torch.cat((normalized_images_matrix, nn.functional.normalize(images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]), p=2, dim=1)), dim=0)

				outputs = cnn_model(images)
				
				sm_outputs = softmax_fn(outputs)
				#taking minus of entropy puts 
				output_uncertainty = 1 - torch.max(sm_outputs, dim=1)[0]

				output_uncertainty_all = torch.cat((output_uncertainty_all, output_uncertainty))
				inds_chosen_to_rank = inds_chosen_to_rank + tl_ind.tolist()

			uncertainty_density_to_rank = torch.mean(torch.mm(normalized_images_matrix, normalized_images_matrix.transpose(dim0=0, dim1=1)), dim=0) * output_uncertainty_all
			if (len(inds_chosen_to_rank) <= num_train_samples_per_step):
				best_sample_indices = inds_chosen_to_rank
			else:
				#Now find the worst 'num_train_samples_per_step' and send the corresponding 'inds_chosen_to_rank' to best_sample_indices
				uncertainty_density_to_rank = np.array(uncertainty_density_to_rank.tolist())
				inds_chosen_to_rank = np.array(inds_chosen_to_rank)
				best_sample_indices = inds_chosen_to_rank[uncertainty_density_to_rank.argsort()[-num_train_samples_per_step:]].tolist()

		return best_sample_indices



class uncertainty_density_entropy_softmax():
	def __init__(self):
		return
	#And by best we mean the ones that give the lowest confidence in the network prediction
	def get_best_sample_indices(self, device, cnn_model, leftover_train_loader, num_train_samples_per_step, num_samples_to_rank):
		best_sample_indices = []
		
		softmax_fn = nn.Softmax()
		cnn_model.eval()
		
		with torch.no_grad():
			
			inds_chosen_to_rank = []
			output_uncertainty_all = torch.tensor([]).to(device)
			normalized_images_matrix = torch.tensor([]).to(device)

			for i, (images, labels, tl_ind) in enumerate(leftover_train_loader):
				if (i*leftover_train_loader.batch_size >= num_samples_to_rank):
					break
				images = images.to(device)
				normalized_images_matrix = torch.cat((normalized_images_matrix, nn.functional.normalize(images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]), p=2, dim=1)), dim=0)

				outputs = cnn_model(images)
				
				sm_outputs = softmax_fn(outputs)
				output_uncertainty =  -torch.sum(sm_outputs * torch.log(sm_outputs), dim=1)

				output_uncertainty_all = torch.cat((output_uncertainty_all, output_uncertainty))
				inds_chosen_to_rank = inds_chosen_to_rank + tl_ind.tolist()

			uncertainty_density_to_rank = torch.mean(torch.mm(normalized_images_matrix, normalized_images_matrix.transpose(dim0=0, dim1=1)), dim=0) * output_uncertainty_all
			
			if (len(inds_chosen_to_rank) <= num_train_samples_per_step):
				best_sample_indices = inds_chosen_to_rank
			else:
				#Now find the worst 'num_train_samples_per_step' and send the corresponding 'inds_chosen_to_rank' to best_sample_indices
				uncertainty_density_to_rank = np.array(uncertainty_density_to_rank.tolist())
				inds_chosen_to_rank = np.array(inds_chosen_to_rank)
				best_sample_indices = inds_chosen_to_rank[uncertainty_density_to_rank.argsort()[-num_train_samples_per_step:]].tolist()

		return best_sample_indices



# class k_means_clustering_sampling():
# 	def __init__(self):
# 		return
# 	#And by best we mean the ones that give the lowest confidence in the network prediction
# 	def get_best_sample_indices(self, device, cnn_model, leftover_train_loader, num_train_samples_per_step, num_samples_to_rank):
# 		best_sample_indices = []
		
# 		softmax_fn = nn.Softmax()
# 		cnn_model.eval()
		
# 		with torch.no_grad():
			
# 			inds_chosen_to_rank = []
# 			confidence_chosen_to_rank = torch.tensor([])

# 			for i, (images, labels, tl_ind) in enumerate(leftover_train_loader):
# 				if (i*leftover_train_loader.batch_size >= num_samples_to_rank):
# 					break
# 				images = images.to(device)
# 				outputs = cnn_model(images)
				
# 				sm_outputs = softmax_fn(outputs)
# 				#taking minus of entropy puts 
# 				output_confidence = torch.sum(sm_outputs * torch.log(sm_outputs)).reshape(1,)
# 				output_confidence = output_confidence.cpu()

# 				confidence_chosen_to_rank = torch.cat((confidence_chosen_to_rank, output_confidence))
# 				inds_chosen_to_rank = inds_chosen_to_rank + tl_ind.tolist()

			
# 			if (len(inds_chosen_to_rank) <= num_train_samples_per_step):
# 				best_sample_indices = inds_chosen_to_rank
# 			else:
# 				#Now find the worst 'num_train_samples_per_step' and send the corresponding 'inds_chosen_to_rank' to best_sample_indices
# 				confidence_chosen_to_rank = np.array(confidence_chosen_to_rank.tolist())
# 				inds_chosen_to_rank = np.array(inds_chosen_to_rank)
# 				best_sample_indices = inds_chosen_to_rank[confidence_chosen_to_rank.argsort()[0:num_train_samples_per_step]].tolist()

# 		return best_sample_indice

# class ActiveLearning():
# 	def __init__

# 	def get_best_sample_indices
