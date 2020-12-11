import torch
from torch.autograd import Variable
import torchvision.models as models
import cv2
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import dataset
# from prune import *
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
    # model = nn.Sequential(*modules)
		self.features = nn.Sequential(*modules)
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

model = ModifiedResNet18Model()

print(model.features._modules.items()[4][1])