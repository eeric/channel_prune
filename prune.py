import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import math
 
def replace_layers(model, i, indexes, layers):
	if i in indexes:
		return layers[indexes.index(i)]
	return model[i]

def prune_resnet18_conv_layer(model, layer_index, filter_index):
	
	next_conv = None
	next_new_conv = None
	downin_conv = None
	downout_conv = None
	next_downin_conv = None
	new_down_conv = None
	if layer_index == 0:
		_, conv = model.features._modules.items()[layer_index]
		next_conv =  model.features._modules.items()[4][1][0].conv1

	if layer_index > 0 and layer_index < 5:
                tt=1
                kt=layer_index//3
                pt=layer_index%2
                if pt==1:
                        conv = model.features._modules.items()[3+tt][1][kt].conv1
                        next_conv =  model.features._modules.items()[3+tt][1][kt].conv2
                else:   
                        if kt==0:
                                conv = model.features._modules.items()[3+tt][1][kt].conv2
                                next_conv =  model.features._modules.items()[3+tt][1][kt+1].conv1
                        else:
                                conv = model.features._modules.items()[3+tt][1][kt].conv2
				next_conv =  model.features._modules.items()[3+tt+1][1][0].conv1
				downin_conv =  model.features._modules.items()[3+tt+1][1][0].downsample[0]

	if layer_index > 4 and layer_index < 9:
                tt=2
                kt=(layer_index-(tt-1)*4)//3
                pt=(layer_index-(tt-1)*4)%2
                if pt==1:
                        conv = model.features._modules.items()[3+tt][1][kt].conv1
                        next_conv =  model.features._modules.items()[3+tt][1][kt].conv2
			#downout_conv =  model.features._modules.items()[3+tt][1][0].downsample[0]
                else:
                        if kt==0:
                                conv = model.features._modules.items()[3+tt][1][kt].conv2
                                next_conv =  model.features._modules.items()[3+tt][1][kt+1].conv1
				downout_conv =  model.features._modules.items()[3+tt][1][kt].downsample[0]
                        else:
                                conv = model.features._modules.items()[3+tt][1][kt].conv2
                                next_conv =  model.features._modules.items()[3+tt+1][1][0].conv1
				downin_conv =  model.features._modules.items()[3+tt+1][1][0].downsample[0]

	if layer_index > 8 and layer_index < 13:
                tt=3
                kt=(layer_index-(tt-1)*4)//3
                pt=(layer_index-(tt-1)*4)%2
                if pt==1:
                        conv = model.features._modules.items()[3+tt][1][kt].conv1
                        next_conv =  model.features._modules.items()[3+tt][1][kt].conv2
			#downout_conv =  model.features._modules.items()[3+tt][1][0].downsample[0]
                else:
                        if kt==0:
                                conv = model.features._modules.items()[3+tt][1][kt].conv2
                                next_conv =  model.features._modules.items()[3+tt][1][kt+1].conv1
				downout_conv =  model.features._modules.items()[3+tt][1][kt].downsample[0]
                        else:
                                conv = model.features._modules.items()[3+tt][1][kt].conv2
                                next_conv =  model.features._modules.items()[3+tt+1][1][0].conv1
				downin_conv =  model.features._modules.items()[3+tt+1][1][0].downsample[0]

	if layer_index > 12 and layer_index < 17:
		tt=4 
		kt=(layer_index-(tt-1)*4)//3
		pt=(layer_index-(tt-1)*4)%2
		if pt==1:
			conv = model.features._modules.items()[3+tt][1][kt].conv1
			next_conv =  model.features._modules.items()[3+tt][1][kt].conv2
		else:
			if kt==0:				
				conv = model.features._modules.items()[3+tt][1][kt].conv2
				next_conv =  model.features._modules.items()[3+tt][1][kt+1].conv1
				downout_conv =  model.features._modules.items()[3+tt][1][kt].downsample[0]
			else:
				conv = model.features._modules.items()[3+tt][1][kt].conv2
				#next_conv =  model.features._modules.items()[5+1][1][0].conv1

	#while layer_index + offset <  len(model.features._modules.items()):
		#res =  model.features._modules.items()[layer_index+offset]
		#if isinstance(res[1], torch.nn.modules.conv.Conv2d) or isinstance(res[1], torch.nn.BatchNorm2d):
		#	next_name, next_conv = res
		#	break
		#offset = offset + 1
	
	new_conv = \
		torch.nn.Conv2d(in_channels = conv.in_channels, \
			out_channels = conv.out_channels - 1,
			kernel_size = conv.kernel_size, \
			stride = conv.stride,
			padding = conv.padding,
			dilation = conv.dilation,
			groups = conv.groups,
			bias = conv.bias)

	old_weights = conv.weight.data.cpu().numpy()
	new_weights = new_conv.weight.data.cpu().numpy()

	new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
	new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
	new_conv.weight.data = torch.from_numpy(new_weights).cuda()

	#bias_numpy = conv.bias.data.cpu().numpy()

	#bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
	#bias[:filter_index] = bias_numpy[:filter_index]
	#bias[filter_index : ] = bias_numpy[filter_index + 1 :]
	#new_conv.bias = torch.from_numpy(bias).cuda()
	if not downout_conv is None:
		new_down_conv = \
			torch.nn.Conv2d(in_channels = downout_conv.in_channels, \
				out_channels = downout_conv.out_channels - 1,
				kernel_size = downout_conv.kernel_size, \
				stride = downout_conv.stride,
				padding = downout_conv.padding,
				dilation = downout_conv.dilation,
				groups = downout_conv.groups,
				bias = downout_conv.bias)

		old_weights = downout_conv.weight.data.cpu().numpy()
		new_weights = new_down_conv.weight.data.cpu().numpy()

		new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
		new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
		new_down_conv.weight.data = torch.from_numpy(new_weights).cuda()

	if not next_conv is None:
		next_new_conv = \
			torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
				out_channels =  next_conv.out_channels, \
				kernel_size = next_conv.kernel_size, \
				stride = next_conv.stride,
				padding = next_conv.padding,
				dilation = next_conv.dilation,
				groups = next_conv.groups,
				bias = next_conv.bias)

		old_weights = next_conv.weight.data.cpu().numpy()
		new_weights = next_new_conv.weight.data.cpu().numpy()

		new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
		new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
		next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

		#next_new_conv.bias = next_conv.bias

	if not downin_conv is None:
		next_downin_conv = \
			torch.nn.Conv2d(in_channels = downin_conv.in_channels - 1,\
				out_channels =  downin_conv.out_channels, \
				kernel_size = downin_conv.kernel_size, \
				stride = downin_conv.stride,
				padding = downin_conv.padding,
				dilation = downin_conv.dilation,
				groups = downin_conv.groups,
				bias = downin_conv.bias)

		old_weights = downin_conv.weight.data.cpu().numpy()
		new_weights = next_downin_conv.weight.data.cpu().numpy()

		new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
		new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
		next_downin_conv.weight.data = torch.from_numpy(new_weights).cuda()

	if not next_conv is None:
		if layer_index ==0:
	 		features1 = torch.nn.Sequential(
	            	*(replace_layers(model.features, i, [layer_index, layer_index], \
	            		[new_conv, new_conv]) for i, _ in enumerate(model.features)))
			del model.features
			model.features = features1
			model.features._modules.items()[4][1][0].conv1 = next_new_conv
		else:
                	if pt==1:
				model.features._modules.items()[3+tt][1][kt].conv1 = new_conv
				model.features._modules.items()[3+tt][1][kt].conv2 = next_new_conv
				#if tt > 1:
					#ds = torch.nn.Sequential(
						#*(replace_layers(model.features._modules.items()[3+tt][1][0].downsample, i, [0], \
							#[new_down_conv]) for i, _ in enumerate(model.features._modules.items()[3+tt][1][0].downsample)))
					#model.features._modules.items()[3+tt][1][kt].downsample = ds
					#model.features._modules.items()[3+tt][1][kt].downsample[0] = next_downin_conv
                	else:   
				if kt==0:                                
					model.features._modules.items()[3+tt][1][kt].conv2 = new_conv
					model.features._modules.items()[3+tt][1][kt+1].conv1 = next_new_conv
					if tt > 1:
						ds = torch.nn.Sequential(
							*(replace_layers(model.features._modules.items()[3+tt][1][kt].downsample, i, [0], \
								[new_down_conv]) for i, _ in enumerate(model.features._modules.items()[3+tt][1][kt].downsample)))
						model.features._modules.items()[3+tt][1][kt].downsample = ds
				else:                                
					model.features._modules.items()[3+tt][1][kt].conv2 = new_conv
					model.features._modules.items()[3+tt+1][kt][0].conv1 = next_new_conv
					#if tt == 1:
					ds = torch.nn.Sequential(
						*(replace_layers(model.features._modules.items()[3+tt+1][kt][0].downsample, i, [0], \
							[next_downin_conv]) for i, _ in enumerate(model.features._modules.items()[3+tt+1][kt][0].downsample)))
					model.features._modules.items()[3+tt+1][kt][0].downsample = ds
					#model.features._modules.items()[3+tt+1][kt][0].downsample[0] = next_downin_conv
			
	 	#model.features._modules.items()[3+tt+1][kt][0].downsample = ds
	 	del conv

	 	#model.features = features

	else:
		#Prunning the last conv layer. This affects the first linear layer of the classifier.
		model.features._modules.items()[3+tt][1][kt].conv2 = new_conv
	 	#model.features = torch.nn.Sequential(
	            #*(replace_layers(model.features, i, [layer_index], \
	            	#[new_conv]) for i, _ in enumerate(model.features)))
	 	layer_index = 0
	 	old_linear_layer = None
	 	for _, module in model.fc._modules.items():
	 		if isinstance(module, torch.nn.Linear):
	 			old_linear_layer = module
	 			break
	 		layer_index = layer_index  + 1

	 	if old_linear_layer is None:
	 		raise BaseException("No linear laye found in classifier")
		params_per_input_channel = old_linear_layer.in_features / conv.out_channels

	 	new_linear_layer = \
	 		torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
	 			old_linear_layer.out_features)
	 	
	 	old_weights = old_linear_layer.weight.data.cpu().numpy()
	 	new_weights = new_linear_layer.weight.data.cpu().numpy()	 	

	 	new_weights[:, : filter_index * params_per_input_channel] = \
	 		old_weights[:, : filter_index * params_per_input_channel]
	 	new_weights[:, filter_index * params_per_input_channel :] = \
	 		old_weights[:, (filter_index + 1) * params_per_input_channel :]
	 	
	 	new_linear_layer.bias.data = old_linear_layer.bias.data

	 	new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

		fc = torch.nn.Sequential(
			*(replace_layers(model.fc, i, [layer_index], \
				[new_linear_layer]) for i, _ in enumerate(model.fc)))

		del model.fc
		del next_conv
		del conv
		model.fc = fc

	return model

if __name__ == '__main__':
	model = models.vgg16(pretrained=True)
	model.train()

	t0 = time.time()
	model = prune_conv_layer(model, 28, 10)
	print "The prunning took", time.time() - t0
