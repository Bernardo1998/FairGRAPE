import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy
import types
import pandas as pd
from collections import defaultdict
import os
import torch.optim as optim
from joblib import Parallel, delayed


# custom codes
from train_and_val import loss_multi_tasks
from util import make_model, custom_forward_conv2d, custom_forward_conv1d, custom_forward_linear
from dataset import split_image_name, make_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

supported_layers = ['Linear', 'Conv2d', 'Conv1d']

forward_mapping_dict = {
    'Linear': custom_forward_linear,
    'Conv2d': custom_forward_conv2d,
    'Conv1d': custom_forward_conv1d
}

################
# Based on SNIP code from github
################
class Prunner:

    def __init__(self, model, criterion, dataloader, output_cols_each_task=None, save_mask=False):
        self.update_model(model)
        self.criterion = criterion.to(device)
        self.dataloader = dataloader
        self.output_cols_each_task=output_cols_each_task
        self.update_forward_pass()
        self.save_mask = save_mask

    def update_model(self, model):
        self.model = copy.deepcopy(model).to(device)
        self.prun_model = copy.deepcopy(model).to(device)

    def get_model(self):
        return self.model

    def init_mask(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))        

    # Expected mask should be a list of arrays, len = number of prunable layers, same shapes as weights
    def apply_hook(self, masks):
        layers = filter(lambda l: type(l).__name__ in forward_mapping_dict, self.prun_model.modules())
        def apply_masking(mask):
            def hook(weight):
                return weight * mask
            return hook
        for layer, mask in zip(layers, masks):
            assert layer.weight.shape == mask.shape
            layer.weight.data = layer.weight.data * mask
            layer.weight.register_hook(apply_masking(mask))

    def prune(self, prune_cfgs, show_pruned_details=False):  
        masks = self.get_mask(prune_cfgs) # get_mask() need to be implemented by child classes
        if self.save_mask:
            print(type(masks), type(masks[0]))
            mask_np = np.array([m.cpu().numpy() for m in masks])
            np.save("mask.npy",mask_np)
            del mask_np
        self.apply_hook(masks)
        mask_by_layer = {}
        for layer in self.prun_model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(masks.pop(0), requires_grad=False)
                mask_by_layer[layer] = layer.mask
        if show_pruned_details:
            self.print_remain()
        return self.prun_model

    def update_forward_pass(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)

    def variance_scaling_init(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False

    def print_remain(self):
        remain, total = 0, 0
        for name, layer in self.prun_model.named_modules():
            if type(layer).__name__ in forward_mapping_dict:
                remain += torch.sum(layer.mask)
                total += torch.prod(torch.tensor(layer.weight.shape))
                print(name, torch.sum(layer.mask), layer.weight.shape)
        print(remain, total, remain/total)

class Random(Prunner):
    def __init__(self, model, criterion, dataloader, output_cols_each_task, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)

    def get_mask(self, prune_cfgs):
        compression_rate, by_layer = prune_cfgs
        masks = []
        if by_layer:
            for layer in self.prun_model.modules():
                mask = np.random.rand(layer.weight.shape)
                keep_params = int((1 - compression_rate) * math.prod(mask.shape))
                values, _ = torch.topk(mask, keep_params, sorted=True)
                threshold = values[-1]
                masks.append((mask > threshold).int())
        else:
            total_params = 0
            for layer in self.prun_model.modules():
                masks.append(np.random.rand(layer.weight.shape))
                total_params += math.prod(layer.weight.shape)
            keep_params = int((1 - compression_rate) * total_params)
            values, _ = torch.topk(masks, keep_params, sorted=True)
            threshold = values[-1]
            masks = [(mask > threshold).int() for mask in masks]
        return masks

class SNIP(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)
         
    def get_mask(self, prune_cfgs):
        compression_factor, num_batch_sampling,init = prune_cfgs
        if init:
            self.variance_scaling_init()
        grads, grads_list = self.compute_grads(num_batch_sampling)
        keep_params = int((1 - compression_factor) * len(grads))
        values, idxs = torch.topk(grads / grads.sum(), keep_params, sorted=True)
        threshold = values[-1]
        masks = [(grad / grads.sum() > threshold).int() for grad in grads_list]
        return masks

    def compute_grads(self, num_batch_sampling=1):
        moving_average_grads = 0
        for i, (data, labels) in enumerate(self.dataloader):
            if i == num_batch_sampling:
                break
            data, labels = data.to(device), labels.to(device)
            out = self.model(data)
            #labels = labels[:,0]
            loss = loss_multi_tasks(out, labels, self.criterion, self.output_cols_each_task, False)
            self.model.zero_grad()
            loss.backward()
            grads_list = []
            for layer in self.model.modules():
                if type(layer).__name__ in forward_mapping_dict:
                    grads_list.append(torch.abs(layer.mask.grad))
            grads = torch.cat([torch.flatten(grad) for grad in grads_list])
            if i == 0:
                moving_average_grads = grads
                moving_average_grad_list = grads_list
            else:
                moving_average_grads = ((moving_average_grads * i) + grads) / (i + 1)
                moving_average_grad_list = [((mv_avg_grad * i) + grad) / (i + 1)
                                            for mv_avg_grad, grad in zip(moving_average_grad_list, grads_list)]
        return moving_average_grads, moving_average_grad_list


############
# GraSP code from git
############
class GraSP(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task=None, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)

    def count_total_parameters(self,net):
        total = 0
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                total += m.weight.numel()
        return total

    def count_fc_parameters(self,net):
        total = 0
        for m in net.modules():
            if isinstance(m, (nn.Linear)):
                total += m.weight.numel()
        return total

    def GraSP_fetch_data(self, dataloader, num_classes, samples_per_class, target_col=0):
        datas = [[] for _ in range(num_classes)]
        labels = [[] for _ in range(num_classes)]
        mark = dict()
        dataloader_iter = iter(dataloader)
        
        while True:
            inputs, targets = next(dataloader_iter)
            for idx in range(inputs.shape[0]):
                x, y = inputs[idx:idx+1], targets[idx:idx+1]
                #print(y.shape, target_col, y[0, target_col])
                if isinstance(target_col, int):
                    category = y[0,target_col] 
                else: # The celeba case
                    category = -1
                    for target_i in target_col:
                        label = y[0,target_i].item()
                        # Use this sample if it is positive in one class that does not have enough sample yet
                        if label == 1 and target_i not in mark:
                            category = target_i
                            break
                #print(len(datas[category]))
                category = category.item() if not isinstance(category, int) else category
                if category == -1: # skip since this sample cannot be used
                    continue
                if len(datas[category]) == samples_per_class:
                    #print(category)
                    mark[category] = True
                    continue
                datas[category].append(x)
                labels[category].append(y)
            print(len(mark))
            if len(mark) == num_classes:
                break

        X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels])
        return X, y

    def get_mask(self, prune_cfgs, num_iters=1, T=200, reinit=True, fair_grad = False):
        ratio,target_col, num_classes, samples_per_class= prune_cfgs
        net = self.model
        train_dataloader = self.dataloader
        output_cols_each_task= self.output_cols_each_task
        eps = 1e-10
        keep_ratio = 1-ratio
        old_net = net
        criterion = F.cross_entropy

        net = copy.deepcopy(net)  # .eval()
        net.zero_grad()

        weights = []
        total_parameters = self.count_total_parameters(net)
        fc_parameters = self.count_fc_parameters(net)

        # rescale_weights(net)
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if isinstance(layer, nn.Linear) and reinit:
                    nn.init.xavier_normal(layer.weight)
                weights.append(layer.weight)
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
                

        inputs_one = []
        targets_one = []

        grad_w = None
        for w in weights:
            w.requires_grad_(True)

        print_once = False
        for it in range(num_iters):
            print("(1): Iterations %d/%d." % (it, num_iters))
            inputs, targets = self.GraSP_fetch_data(train_dataloader, num_classes, samples_per_class, target_col)
            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)
            inputs_one.append(din[:N//2])
            targets_one.append(dtarget[:N//2])
            inputs_one.append(din[N // 2:])
            targets_one.append(dtarget[N // 2:])
            inputs = inputs.to(device)
            targets = targets.to(device)


            outputs = net.forward(inputs[:N//2])/T
            if print_once:
                # import pdb; pdb.set_trace()
                x = F.softmax(outputs)
                print(x)
                print(x.max(), x.min())
                print_once = False
            #print(outputs.shape, targets[:N//2].shape)
            loss = loss_multi_tasks(outputs, targets[:N//2], criterion, output_cols_each_task)
            # ===== debug ================
            #print(loss.requires_grad, loss.shape)
            #print(weights.requires_grad, weights.shape)
            grad_w_p = autograd.grad(loss, weights)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

            outputs = net.forward(inputs[N // 2:])/T
            loss = loss_multi_tasks(outputs, targets[N//2:], criterion, output_cols_each_task)
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

        ret_inputs = []
        ret_targets = []

        for it in range(len(inputs_one)):
            print("(2): Iterations %d/%d." % (it, num_iters))
            inputs = inputs_one.pop(0).to(device)
            targets = targets_one.pop(0).to(device)
            ret_inputs.append(inputs)
            ret_targets.append(targets)
            outputs = net.forward(inputs)/T
            loss = loss_multi_tasks(outputs, targets, criterion, output_cols_each_task)
            # ===== debug ==============

            grad_f = autograd.grad(loss, weights, create_graph=True)
            z = 0
            count = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    z += (grad_w[count].data * grad_f[count]).sum()
                    count += 1
            z.backward()

        grads = dict()
        old_modules = list(old_net.modules())
        selected_layers = []
        for idx, (name, layer) in enumerate(net.named_modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grad = -layer.weight.data * layer.weight.grad
                if fair_grad:
                    grad += 0
                grads[old_modules[idx]] = grad  # -theta_q Hg, with possible fairness term
                selected_layers.append(idx)

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
        norm_factor = torch.abs(torch.sum(all_scores)) + eps
        print("** norm factor:", norm_factor)
        all_scores.div_(norm_factor)

        num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
        threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
        # import pdb; pdb.set_trace()
        acceptable_score = threshold[-1]
        print('** accept: ', acceptable_score)
        keep_masks = dict()    
        for m, g in grads.items(): # m is layer and g is the score, smaller means more important
            #print(m)
            keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()
        
        # The code above are from GraSP repo. Below turn the mask into a list for prunner.
        mask_list = []
        for idx, (name, layer) in enumerate(net.named_modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                mask_list.append(keep_masks[old_modules[idx]])
            elif type(layer).__name__ in forward_mapping_dict:
                mask_list.append(layer.mask)

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

        return mask_list

######################
# Deep compression code
# If s is a number, then all layers would have the same sensitivity
# If s is a dict then each layer could have its own sensitivity
#######################
class WS(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task=None, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)

    def get_mask(self, prune_cfgs):
        net = self.model
        pruning_rate=prune_cfgs[0]# Larger number means more are pruned
        masks = []
        for name, module in net.named_modules():
            if type(module).__name__ not in forward_mapping_dict:
                continue
            s = pruning_rate[name] if isinstance(pruning_rate, dict) else pruning_rate
            n_to_kepp = int(torch.prod(torch.tensor(module.weight.shape)) * (1-s))
            threshold = torch.topk(module.weight.data.cpu().view(-1).abs(), n_to_kepp, sorted=True)[0][-1]
            #print(f'Weight Pruning with threshold : {threshold} for layer {name}')
            mask = np.where(abs(module.weight.cpu().abs()) <= threshold, 0, 1)
            mask = torch.tensor(mask).to(device)
            masks.append(mask)

        return masks

######################
# Lottery selection
# save the initial state, reset to it after each pruning
#######################
class Lottery(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task=None, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)
        self.initial_model = copy.deepcopy(model)

    def get_mask(self, prune_cfgs):
        net = self.model
        pruning_rate=prune_cfgs[0]# Larger number means more are pruned
        masks = []
        for name, module in net.named_modules():
            if type(module).__name__ not in forward_mapping_dict:
                continue
            s = pruning_rate[name] if isinstance(pruning_rate, dict) else pruning_rate
            n_to_kepp = int(torch.prod(torch.tensor(module.weight.shape)) * (1-s))
            threshold = torch.topk(module.weight.data.cpu().view(-1).abs(), n_to_kepp, sorted=True)[0][-1]
            print(f'Weight Pruning with threshold : {threshold} for layer {name}')
            mask = np.where(abs(module.weight.cpu().abs()) <= threshold, 0, 1)
            mask = torch.tensor(mask).to(device)
            masks.append(mask)
        # Reset model to initial state, unique for Lottery
        self.update_model(self.initial_model)

        return masks

############################
# Importance estimation, w/o sensitive groups
############################
class Importance(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)
        self.init_mask()

    def init_mask(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))

    def get_mask(self, prune_cfgs):
        prune_ratio, test_csv, new_img_dir, _, masked_grads, output_cols_each_task ,col_used, _,_, stop_batch, _ = prune_cfgs
        masks = []
        _,impts = importance_by_class0(self.model, test_csv, new_img_dir, masked_grads,output_cols_each_task,col_used,stop_batch=stop_batch)
        for name,layer in self.model.named_modules():
            if name not in impts[0]:
                continue
            impt = impts[0][name]
            keep_params = int((1 - prune_ratio) * math.prod(impt.shape))
            print(name, impt.shape, prune_ratio, keep_params)
            values, _ = torch.topk(impt.view(-1), keep_params, sorted=True)
            threshold = values[-1]
            masks.append((impt  > threshold).int().to(device))
        return masks

############################
# Fairness selection
############################
class FairGRAPE(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)
        self.init_mask()

    def init_mask(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))

    def get_mask(self, prune_cfgs):
        prune_ratio, test_csv, new_img_dir, sensitive_classes, masked_grads, output_cols_each_task ,col_used, para_batch, impt_type, stop_batch, delta_p = prune_cfgs
        print("Sensitive classes:",sensitive_classes)
        mask = fairness_grad(self.model, prune_ratio, test_csv, new_img_dir, sensitive_classes, masked_grads, output_cols_each_task ,col_used, para_batch, impt_type, stop_batch, delta_p)
        return mask

def fairness_grad(model, prune_ratio, test_csv, new_img_dir=None, sensitive_classes = 2, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)],col_names=['race','gender'], para_batch=1, impt_type = 0, stop_batch=10000, delta_p=False,n_jobs=1):

	if impt_type == 0:
		_,grad_mag_by_race = importance_by_class0(model, test_csv, new_img_dir=new_img_dir, masked_grads=masked_grads, output_cols_each_task=output_cols_each_task,col_names=col_names,stop_batch=stop_batch)
	elif impt_type == 1:
		_,grad_mag_by_race = importance_by_class1(model, test_csv, new_img_dir=new_img_dir, masked_grads=masked_grads, output_cols_each_task=output_cols_each_task,col_names=col_names, n_classes=sensitive_classes)	
	elif impt_type == 2:
		_,grad_mag_by_race = importance_by_class2(model, test_csv, new_img_dir, output_cols_each_task,col_names)	

	# calculate the target distribution of gradient on pre-pruning model at each layer
	# Note that this input model might have been previously pruned as well.
	grad_mag_each_race = defaultdict(list)
	for race in grad_mag_by_race.keys():
		race_grad_mag = grad_mag_by_race[race]
		for layer_name in race_grad_mag:
			grad_mag_each_race[layer_name].append(torch.sum(race_grad_mag[layer_name].abs()))

        # CAUTION, grads still have negatives.
	n_classes = sensitive_classes
	grads_by_race_merged = make_mask_by_grad(grad_mag_by_race,n_classes)
	unpruned_grad = grad_mag_each_race
	grad_target, grad_target_total = {}, np.array([0.0] * n_classes)
	for layer_name in unpruned_grad:
		grad_this_layer = unpruned_grad[layer_name]
		grad_target[layer_name] = np.array([grad/sum(grad_this_layer) for grad in grad_this_layer])
		#print(np.array(grad_this_layer))
		grad_target_total += np.array(grad_this_layer)

	# For each weight, record its group-wise importance and idx within layer.
	# Notice! The selection below can be done using different metrics
	grad_by_layer_sorted = {}
	for name, layer in model.named_modules():
		grad_by_layer_sorted[name] = {}
		if type(layer).__name__ not in supported_layers:
			continue
		selected = layer.mask
		idxs,idxs_tp = selected.nonzero(), selected.nonzero(as_tuple=True)
		grad_this_layer = grads_by_race_merged[name][idxs_tp]
		sum_per_node = torch.sum(grad_this_layer, 1)
		for race in range(n_classes):
			race_col = grad_this_layer[:, race]
			# While a node might have high importance for one race, it might also have even 
			# larger importance for another, which ultimately decreases share for this race.
			if delta_p == 1:
				race_col /= sum_per_node
			elif delta_p == 2:
				race_col *= (race_col / sum_per_node)
			_, sorted_idx = torch.topk(race_col, k = len(race_col), sorted=True)
			grad_by_layer_sorted[name][race] = [grad_this_layer[sorted_idx], idxs[sorted_idx]]

	####################
	# greedy method
	####################
	mask_list = []
	# record how many weights to select at each layer, use for layer wise connection
	nodes_each_layer = {}
	for i in range(n_classes):
		nodes_each_layer[i] = []

	layer_parameters = []
	for name,layer in model.named_modules():
		layer_parameters.append([name, layer,grad_by_layer_sorted[name],grad_target,n_classes])

	# This function is designed to facilitate parallel processing, but only n_jobs = 1 available for now.
	def greed_one_layer(layer_parameter):
		name, layer,grad_by_layer_sorted_layer,grad_target,n_classes = layer_parameter
		if type(layer).__name__ not in supported_layers:
			return {name:None}
		print("Performing greedy selection on {}".format(name))
		mask_this_layer = torch.zeros(layer.weight.shape)
		layer_total = int(torch.prod(torch.tensor(layer.weight.shape)))
		num_to_select_this_layer = int(layer_total * (1-prune_ratio))

		n_selected_this_layer = 0
		last_printed_freq = 0

		grad_target_this_layer = grad_target[name]
		grads_by_race_selected = np.array([0] * n_classes, dtype=float)
		grads_prop_by_race = np.array([1/n_classes] * n_classes, dtype=float)

		grads_by_race_idx = np.array([0] * n_classes)
		last_race_updated = 0
		while n_selected_this_layer < num_to_select_this_layer:
			# find the race that currently has the larget deficient
			race_diff = grads_prop_by_race - grad_target_this_layer
			if last_race_updated == 0:
				race_to_add = race_diff.argmin()
			last_race_updated = last_race_updated + 1 if last_race_updated < para_batch else 0
			idx_in_seq = grads_by_race_idx[race_to_add]
			# grads here are already abs
			grads = grad_by_layer_sorted_layer[race_to_add][0][idx_in_seq]
			idx = tuple(grad_by_layer_sorted_layer[race_to_add][1][idx_in_seq])
			selected_condition = mask_this_layer[idx]
			# only add weights that have neer been selected
			if selected_condition == 0:
				n_selected_this_layer += 1
				grads_by_race_selected += grads.cpu().numpy()
				grads_prop_by_race = grads_by_race_selected / sum(grads_by_race_selected)
				mask_this_layer[idx] = 1
			grads_by_race_idx[race_to_add] += 1
		
		return {name:mask_this_layer}

	names_and_masks = Parallel(n_jobs=n_jobs)(delayed(greed_one_layer)(lp) for lp in layer_parameters)

	mask_by_layernames = dict([pair for d in names_and_masks for pair in d.items()])
	mask_list = [mask_by_layernames[name].to(device) for name,layer in model.named_modules() if type(layer).__name__ in supported_layers]

	return mask_list

# The last col in the label matrix is for sensitive attr, others for non-protected ones
# This order of label is given by col_names, the last one is the sensitive group.
def importance_by_class0(model_path, test_csv, new_img_dir=None, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'],network=None,optimizer=None, lr=1e-4, stop_batch=10000):
    supported_layers = ['Linear', 'Conv2d', 'Conv1d']

    # Load pruned and retrained model
    model = model_path 

    model.train()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    test_frame = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv
    criterion = nn.CrossEntropyLoss()
    criterion_sensitive = nn.BCELoss()
    activation = nn.Sigmoid()

    # Make sure all images in test frame exist
    if new_img_dir:
        initial_rows = test_frame.shape[0]
        faces = set(os.listdir(new_img_dir))
        faces_found = 0
        new_face_name = []
        face_found_mask = []
        for i in range(test_frame.shape[0]):
            face_name_align = split_image_name(test_frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                faces_found += 1
                new_face_name.append(os.path.join(new_img_dir, face_name_align))
        test_frame = test_frame[face_found_mask].reset_index(drop=True)
        test_frame['face_name_align'] = new_face_name
    test_loader,_ =  make_datasets(test_frame,test_frame,True,64,col_used=col_names)

    model.train()
    sensitive_cols_in_target = len(output_cols_each_task)
    sensitive_groups = sorted(set(test_frame[col_names[-1]]))

    # do mini-batches to get results
    grad_each_group = {}
    H_each_group = {}
    mask_at_each_layer = {}
    batches = 0
    for batch_idx, sample_batched in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
        batches += 1
        if batch_idx % 200 == 0:
            print("{}th mini-batch of importance!".format(batch_idx))
        image_batched, label_batched = sample_batched
        image_batched = image_batched.to(device, dtype=torch.float)
        # transfer it all to gpu
        label_batched = label_batched.to(device)
        for group_idx, group in enumerate(sensitive_groups):
           gradients = {}
           hessians = {}
           # calculate non-protected loss for this group only 
           obs_this_group = torch.squeeze((label_batched[:, sensitive_cols_in_target] == group).nonzero())
           outputs = model(image_batched)
           output_cols_for_non_protected = output_cols_each_task[:(len(output_cols_each_task))]
           outputs_this_group = outputs[obs_this_group,:].view(-1,outputs.shape[1])
           if outputs_this_group.shape[0] < 1 or len(outputs_this_group.shape) < 2:
               continue
           targets_this_group = label_batched[obs_this_group,:].view(-1, label_batched.shape[1])
           loss_non_protected = loss_multi_tasks(outputs_this_group,targets_this_group,criterion,output_cols_for_non_protected)
           loss = loss_non_protected

           loss.backward()
           optimizer.step()

           # get and save all gradient for this group
           for name, layer in model.named_modules():
                if type(layer).__name__ in supported_layers:
                    grads = layer.weight.grad.clone().detach().cpu()
                    weights = layer.weight.data.clone().detach().cpu()
                    # Confirm the model is actually pruned
                    if masked_grads:
                        masks = layer.mask.clone().detach().cpu()
                        mask_at_each_layer[name] = [torch.sum(masks), grads.shape]
                        grads *= masks
                    hessians[name] = (weights.abs() * grads.abs())**2
                    gradients[name] = grads
           if group_idx not in grad_each_group:
               grad_each_group[group_idx] = copy.deepcopy(gradients)
               H_each_group[group_idx] = copy.deepcopy(hessians)
           else:
               for name, layer in model.named_modules():
                   if type(layer).__name__ in supported_layers:
                       grad_each_group[group_idx][name] += gradients[name]
                       H_each_group[group_idx][name] += hessians[name]

    for name, layer in model.named_modules():
       if type(layer).__name__ in supported_layers:
           grad_each_group[group_idx][name] /= batches
           H_each_group[group_idx][name] /= batches
                     
    return grad_each_group, H_each_group

def importance_by_class1(model_path, test_csv, new_img_dir=None, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'],network=None,sample_per_class=32,optimizer=None, lr=1e-4, n_classes=2):
    supported_layers = ['Linear', 'Conv2d', 'Conv1d']
    model = model_path 
    
    test_frame = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv
    criterion = nn.CrossEntropyLoss()
    criterion_sensitive = nn.BCELoss()
    activation = nn.Sigmoid()

    # Make sure all images in test frame exist
    if new_img_dir:
        initial_rows = test_frame.shape[0]
        faces = set(os.listdir(new_img_dir))
        faces_found = 0
        new_face_name = []
        face_found_mask = []
        for i in range(test_frame.shape[0]):
            face_name_align = split_image_name(test_frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                faces_found += 1
                new_face_name.append(os.path.join(new_img_dir, face_name_align))
        test_frame = test_frame[face_found_mask].reset_index(drop=True)
        test_frame['face_name_align'] = new_face_name
    test_loader,_ =  make_datasets(test_frame,test_frame,True,64,col_used=col_names)

    # Select a random batch of image
    model.train()
    test_loader = iter(test_loader)
    sensitive_cols_in_target = len(output_cols_each_task)
    images, targets, comb_idx = fetch_a_fair_batch(test_loader, n_classes, sample_per_class, sensitive_cols_in_target)
    targets = targets.to(device)
    sensitive_groups = sorted([[int(i) for i in comb.split('_')] for comb in comb_idx])
    sensitive_group_idx_in_output = [i for i in range(len(sensitive_groups))]

    # do one forward pass to get the gradient
    outputs = torch.squeeze(model(images.to(device)))

    grad_each_group = {}
    H_each_group = {}
    mask_at_each_layer = {}
    for group_idx, group in enumerate(sensitive_groups):
       gradients = {}
       hessians = {}
       # calculate non-protected loss for this group only 
       obs_this_group = torch.squeeze((targets[:, sensitive_cols_in_target] == group[0]).nonzero())
       output_cols_for_non_protected = output_cols_each_task[:(len(output_cols_each_task))]
       outputs_this_group = outputs[obs_this_group,:]
       targets_this_group = targets[obs_this_group,:]
       loss_non_protected = loss_multi_tasks(outputs_this_group,targets_this_group,criterion,output_cols_for_non_protected)
       # Add sensitive group loss for this group only
       cur_sensitive_group_output = outputs[obs_this_group, sensitive_group_idx_in_output[group_idx]].to(device)
       sensitive_target_this_group = torch.squeeze(targets[obs_this_group, sensitive_cols_in_target] == group[0]).clone().float().to(device)
       loss = loss_non_protected + criterion_sensitive(activation(cur_sensitive_group_output).view(-1), sensitive_target_this_group).cuda()
       loss = loss_non_protected

       try:
           loss.backward(retain_graph=True)
       except:
           print(loss)
           pass

       # get and save all gradient for this group
       for name, layer in model.named_modules():
            if type(layer).__name__ in supported_layers:
                grads = layer.weight.grad.clone().detach().cpu()
                weights = layer.weight.data.clone().detach().cpu()
                # Confirm the model is actually pruned
                if masked_grads:
                    masks = layer.mask.clone().detach().cpu()
                    mask_at_each_layer[name] = [torch.sum(masks), grads.shape]
                    grads *= masks
                hessians[name] = (weights.abs() * grads.abs())**2
                gradients[name] = grads
       grad_each_group[group_idx] = copy.deepcopy(gradients)
       H_each_group[group_idx] = copy.deepcopy(hessians)
                     
    return grad_each_group, H_each_group

# Keys of grad_each_group are sensitive groups.
# Keys of grad_at_each_layer are model layers.
def make_mask_by_grad(grad_each_group, n_classes=7):
    groups = [i for i in range(n_classes)]
    layer_names = list(grad_each_group[0].keys())
    grad_at_each_layer = {}
    for layer in layer_names:
        layer_shape = tuple(list(grad_each_group[groups[0]][layer].shape)+[1])
        grad_merged = torch.cat([grad_each_group[group][layer].view(layer_shape) for group in groups], dim=len(layer_shape)-1)
        grad_at_each_layer[layer] = grad_merged
    return grad_at_each_layer

# From GraSP github:
def fetch_a_fair_batch(dataloader, num_classes, samples_per_class, target_col):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    # combination_idx returns a dict whose keys are sensitive group names in strs
    combination_idx = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y[0,target_col].item()
            #print(target_col, num_classes,category)
            combination_idx[str(category)] = category
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels])
    return X, y, combination_idx


def importance_by_class2(model_path, test_csv, new_img_dir=None, output_cols = [(0,7)], col_names=['race'],masked_grads=True,sample_per_class=10,lr=1e-5):
    supported_layers = ['Linear', 'Conv2d', 'Conv1d']

    # Load pruned and retrained model
    model = model_path 

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    test_frame = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv
    criterion = nn.BCELoss()
    criterion_sensitive = nn.BCELoss()
    activation = nn.Sigmoid()

    # Make sure all images in test frame exist
    if new_img_dir:
        initial_rows = test_frame.shape[0]
        faces = set(os.listdir(new_img_dir))
        faces_found = 0
        new_face_name = []
        face_found_mask = []
        for i in range(test_frame.shape[0]):
            face_name_align = split_image_name(test_frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                faces_found += 1
                new_face_name.append(os.path.join(new_img_dir, face_name_align))
        test_frame = test_frame[face_found_mask].reset_index(drop=True)
        test_frame['face_name_align'] = new_face_name

    test_loader,_ =  make_datasets(test_frame,test_frame,True,64,col_used=col_names)

    model.train()

    # Select a random batch of image
    # num_classes is output classes, not sensitive classes
    test_loader = iter(test_loader)
    target_col = 0
    num_classes = output_cols[0][1] - output_cols[0][0]
    images, targets, comb_idx = fetch_a_fair_batch(test_loader, num_classes, sample_per_class,target_col) # changed 1031

    # do one forward pass to get the gradient
    outputs = torch.squeeze(model(images.to(device)))

    # There are always two columns in the target, one for output and one for sensitive group (0516)
    group_outputs = outputs
    group_targets = torch.squeeze(targets[:, target_col]).to(device)

    # sorting ensures the classes are in the correct orders
    groups = sorted([[int(i) for i in comb.split('_')] for comb in comb_idx])
    grad_each_group = {}
    H_each_group = {}
    mask_at_each_layer = {}

    for group_idx, group in enumerate(groups):
       gradients = {}
       hessians = {}
       output_this_group = group_outputs[:, group]
       target_this_group = (group_targets == group[0]).clone().detach().float()
       loss = criterion(activation(output_this_group).view(-1), target_this_group)
       try:
           loss.backward(retain_graph=True)
       except:
           print(loss)
           pass
       # get and save all gradient for this group
       for name, layer in model.named_modules():
            if type(layer).__name__ in supported_layers:
                grads = layer.weight.grad.clone().detach().cpu()
                weights = layer.weight.data.clone().detach().cpu()
                # Confirm the model is actually pruned
                if masked_grads:
                    masks = layer.mask.clone().detach().cpu()
                    mask_at_each_layer[name] = [torch.sum(masks), grads.shape]
                    grads *= masks
                hessians[name] = weights.abs() * grads.abs()
                gradients[name] = grads
       grad_each_group[group_idx] = copy.deepcopy(gradients)
       H_each_group[group_idx] = copy.deepcopy(hessians)

                     
    return grad_each_group, H_each_group

def save_impt_df(cfgs):
    best_model, test_csv, new_img_dir, masked_grads,output_cols_each_task,col_used,stop_batch = cfgs
    _,impts = importance_by_class0(best_model, test_csv, new_img_dir, masked_grads,output_cols_each_task,col_used,stop_batch=stop_batch)
    impt_df = {}
    n_groups = len(impts)
    for i in range(n_groups):
        impt_df["".join(['group', str(i)])] = []
        
    for name, layer in best_model.named_modules():
        for i in range(n_groups):
            impt_df["".join(['group', str(i)])].append(impts[i][name].sum())
            
    impt_df = pd.DataFrame(impt_df)
    impt_df.to_csv("importance_by_layer.csv")
