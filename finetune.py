import torch
from torch.autograd import Variable
import torchvision.models as models
import cv2
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time


class ModifiedResNet18Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedResNet18Model, self).__init__()

		model = models.resnet18(pretrained=True)#squeezenet1_1
		#model = torch.load('/home/yq/work/face_class/id_rec_resnet_copy/id_rec_resnet/logs/resnet18-1/model.bin')
		modules = list(model.children())[:-1]      # delete the last fc layer.
                model = nn.Sequential(*modules)
		self.features = model
		print("start pruning:")
		for param in self.features.parameters():
			param.requires_grad = False

		self.fc = nn.Sequential(
			#nn.Linear(512, 100)
			nn.Dropout(),
			nn.Linear(512,400),
			nn.ReLU(inplace=True),
                        nn.Dropout(),
			nn.Linear(400,256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 100))
		#modules = list(resnet.children())[:-1]      # delete the last fc layer.
		#resnet = nn.Sequential(*modules)
		#self.classifier = nn.Sequential(
		#    nn.Dropout(),
		#    nn.Linear(512, 256),#25088
		#    nn.ReLU(inplace=True),
		    #nn.Dropout(),
		    #nn.Linear(2048, 2048),
		    #nn.ReLU(inplace=True),
		#    nn.Linear(256, 100))
		#self.features.fc = self.classifier

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)#self.classifier(x)
		return x

class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.reset()
	
	def reset(self):
		# self.activations = []
		# self.gradients = []
		# self.grad_index = 0
		# self.activation_to_layer = {}
		self.filter_ranks = {}

	def forward(self, x):
		self.activations = []
		self.gradients = []
		self.grad_index = 0
		self.activation_to_layer = {}
		#str=['conv1','conv2']
		activation_index = 0
		kk = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
		    if layer < 4 or layer > 7 :		    
			x = module(x)
		    if isinstance(module, torch.nn.modules.conv.Conv2d): #or isinstance(module, torch.nn.BatchNorm2d):
		    	x.register_hook(self.compute_rank)
		        self.activations.append(x)
		        self.activation_to_layer[activation_index] = kk
		    	activation_index += 1
			kk += 1
		    if layer==4 or layer==5 or layer==6 or layer==7:
			for kt in range(2):
				x = self.model.features._modules.items()[layer][1][kt].conv1(x)
				x.register_hook(self.compute_rank)
                        	self.activations.append(x)
                        	self.activation_to_layer[activation_index] = kk
                        	activation_index += 1
				kk += 1
				x = self.model.features._modules.items()[layer][1][kt].bn1(x)
				x = self.model.features._modules.items()[layer][1][kt].relu(x)
				x = self.model.features._modules.items()[layer][1][kt].conv2(x)
                                x.register_hook(self.compute_rank)
                                self.activations.append(x)
                                self.activation_to_layer[activation_index] = kk
                                activation_index += 1
				kk += 1
				x = self.model.features._modules.items()[layer][1][kt].bn2(x)

		return self.model.fc(x.view(x.size(0), -1))

	def compute_rank(self, grad):
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]
		values = \
			torch.sum((activation * grad), dim = 0).\
				sum(dim=2).sum(dim=3)[0, :, 0, 0].data
		
		# Normalize the rank by the filter dimensions
		values = \
			values / (activation.size(0) * activation.size(2) * activation.size(3))

		if activation_index not in self.filter_ranks:
			self.filter_ranks[activation_index] = \
				torch.FloatTensor(activation.size(1)).zero_().cuda()

		self.filter_ranks[activation_index] += values
		self.grad_index += 1

	def lowest_ranking_filters(self, num):
		data = []
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i])
			v = v / np.sqrt(torch.sum(v * v))
			self.filter_ranks[i] = v.cpu()

	def model_forward(self, x):
                for layer, (name, module) in enumerate(self.model.features._modules.items()):
                    if layer < 4 or layer > 7 :
                        x = module(x)
                    else:
                        for kt in range(2):
                                x = self.model.features._modules.items()[layer][1][kt].conv1(x)
                                x = self.model.features._modules.items()[layer][1][kt].bn1(x)
                                x = self.model.features._modules.items()[layer][1][kt].relu(x)
                                x = self.model.features._modules.items()[layer][1][kt].conv2(x)
                                x = self.model.features._modules.items()[layer][1][kt].bn2(x)

                return self.model.fc(x.view(x.size(0), -1))

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)

		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))

		return filters_to_prune				

class PrunningFineTuner_ResNet18:
	def __init__(self, train_path, test_path, model):
		self.train_data_loader = dataset.loader(train_path)
		self.test_data_loader = dataset.test_loader(test_path)

		self.model = model
		self.criterion = torch.nn.CrossEntropyLoss()
		self.prunner = FilterPrunner(self.model) 
		self.model.train()

	def test(self):
		self.model.eval()
		self.model.cuda()
		correct = 0
		total = 0

		for i, (batch, label) in enumerate(self.test_data_loader):
			batch = batch.cuda()
			indata = Variable(batch)
			output = self.prunner.model_forward(indata)
			#output = model(Variable(batch))
			pred = output.data.max(1)[1]
	 		correct += pred.cpu().eq(label).sum()
	 		total += label.size(0)
	 	
	 	print "Accuracy :", str(100*float(correct) / total) + "%"
	 	
	 	self.model.train()

	def train(self, optimizer = None, epoches = 10):
		if optimizer is None:
			optimizer = \
				optim.SGD(model.fc.parameters(), 
					lr=0.0001, momentum=0.9)

		for i in range(epoches):
			print "Epoch: ", i
			self.train_epoch(optimizer)
			self.test()
		print "Finished fine tuning."
		

	def train_batch(self, optimizer, batch, label, rank_filters):
		self.model.zero_grad()
		input = Variable(batch)
		
		if rank_filters:
			#print("good")
			output = self.prunner.forward(input)
			#print(output)
			self.criterion(output, Variable(label)).backward()
		else:
			self.criterion(self.prunner.model_forward(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		for batch, label in self.train_data_loader:
			self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

	def get_candidates_to_prune(self, num_filters_to_prune):
		self.prunner.reset()

		self.train_epoch(rank_filters = True)
		
		self.prunner.normalize_ranks_per_layer()

		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
			if layer < 4 or layer > 7 :
				if isinstance(module, torch.nn.modules.conv.Conv2d):
					filters = filters + module.out_channels
			else:
				for kt in range(2):
					filters = filters + model.features._modules.items()[layer][1][kt].conv1.out_channels
					filters = filters + model.features._modules.items()[layer][1][kt].conv2.out_channels
		return filters

	def batchnorm_modify(self):
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
			if layer < 4 or layer > 7 :
				if isinstance(module, torch.nn.modules.conv.Conv2d):
					conv = torch.nn.BatchNorm2d(num_features=module.out_channels, eps=1e-05, momentum=0.1, affine=True)
					model.features = torch.nn.Sequential(
						*(replace_layers(model.features, i, [layer+1], \
							[conv]) for i, _ in enumerate(model.features)))
			else:
				for kt in range(2):
					conv1 = torch.nn.BatchNorm2d(model.features._modules.items()[layer][1][kt].conv1.out_channels, eps=1e-05, momentum=0.1, affine=True)
					model.features._modules.items()[layer][1][kt].bn1 = conv1
					conv2 = torch.nn.BatchNorm2d(model.features._modules.items()[layer][1][kt].conv2.out_channels, eps=1e-05, momentum=0.1, affine=True)
					model.features._modules.items()[layer][1][kt].bn2 = conv2
					if layer > 4 and layer < 8 and kt ==0:
						convd = torch.nn.BatchNorm2d(model.features._modules.items()[layer][1][kt].conv2.out_channels, eps=1e-05, momentum=0.1, affine=True)
						ds = torch.nn.Sequential(
						*(replace_layers(model.features._modules.items()[layer][1][kt].downsample, i, [1], \
							[convd]) for i, _ in enumerate(model.features._modules.items()[layer][1][kt].downsample)))
						model.features._modules.items()[layer][1][kt].downsample = ds
		return model	
		
	def prune(self):
		#Get the accuracy before prunning
		self.test()

		self.model.train()

		#Make sure all the layers are trainable
		for param in self.model.features.parameters():
			param.requires_grad = True

		number_of_filters = self.total_num_filters()
		#print(number_of_filters)
		num_filters_to_prune_per_iteration = 512
		iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

		iterations = int(iterations * 2.0 / 3)
		#print(iterations)
		print "Number of prunning iterations to reduce 67% filters", iterations

		for _ in range(iterations):
			print "Ranking filters: ", _, "times .."
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1 
			#print "All channels pruned distribution", prune_targets
			print "Layers that will be prunned", layers_prunned
			print "Prunning filters.. "
			model = self.model.cpu()
			for layer_index, filter_index in prune_targets:
				#if layer_index ==14:
					#print "pause.."
				model = prune_resnet18_conv_layer(model, layer_index, filter_index)
			model = self.batchnorm_modify()
			self.model = model.cuda()
			print "Plan to prune...", model

			message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
			print "Filters prunned", str(message)
			#for batch, label in self.train_data_loader:
				#input = Variable(batch.cuda())
				#output = self.prunner.forward(input)
			self.test()
			print "Fine tuning to recover from prunning iteration."
			optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
			self.train(optimizer, epoches = 10)


		print "Finished. Going to fine tune the model a bit more"
		self.train(optimizer, epoches = 4)
		torch.save(model, "prunned_18")
		print("All Pruning End !")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()

	if args.train:
		model = ModifiedResNet18Model().cuda()
	elif args.prune:
		model = torch.load('model_18').cuda()
	if args.train or args.prune:
		print model
	fine_tuner = PrunningFineTuner_ResNet18(args.train_path, args.test_path, model)

	if args.train:
		fine_tuner.train(epoches = 10)
		torch.save(model, "model_18")

	elif args.prune:
		fine_tuner.prune()
