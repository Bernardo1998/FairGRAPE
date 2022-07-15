from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import os
from collections import defaultdict
from dataset import split_image_name
from tqdm import tqdm

import gdown
import zipfile
import tarfile
from statistics import stdev
import shutil

# Initialize with  pretrained model
def make_model(pruning = True, network = "resnet34", dataset="FairFace",n_classes=18):
	if network == "resnet34":
		model_conv = models.resnet34(pretrained=True) if pruning else torchvision.models.resnet34(pretrained=True)
		num_ftrs = model_conv.fc.in_features
		model_conv.fc = nn.Linear(num_ftrs, n_classes)
	elif network == 'mobilenetv2':
		model_conv = models.mobilenet_v2(pretrained=True)
		num_ftrs = model_conv.classifier[1].in_features
		model_conv.classifier[1] = nn.Linear(num_ftrs, n_classes)
	else:
		raise NotImplementedError("{} is not implemented!".format(network))

	return model_conv

def check_imgs(frame, new_img_dir):
	initial_rows = frame.shape[0]
	faces = set(os.listdir(new_img_dir))
	faces_found = 0
	new_face_name = []
	face_found_mask = []
	for i in range(frame.shape[0]):
		face_name_align = split_image_name(frame['face_name_align'][i])
		face_found_mask.append(face_name_align in faces)
		if face_name_align in faces:
			faces_found += 1
			new_face_name.append(os.path.join(new_img_dir, face_name_align))
	frame = frame[face_found_mask].reset_index(drop=True)
	frame['face_name_align'] = new_face_name
	print("{} out of {} faces are found in new dir!".format(faces_found, initial_rows))

	return frame

def check_fairness0(trained_model, test_frame, new_img_dir, n_classes=7, arch="resnet34", col_used=['race'], output_cols_each_task=[(0,7)], sensitive_group="gender"):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# predict on test
	if isinstance(trained_model, str):
		if arch == 'resnet34':
			model = models.resnet34()
			model.fc = nn.Linear(model.fc.in_features, n_classes)
		elif arch == 'mobilenetv2':
			model = models.mobilenet_v2()
			model.classifier = nn.Linear(model.classifier.in_features, n_classes)
		model.load_state_dict(torch.load(trained_model))
	else:
		model = trained_model
	model = model.to(device)
	model.eval()

	trans = transforms.Compose([
	    #transforms.ToPILImage(),
	    transforms.Resize((224, 224)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	    ])
	face_names = []
	scores = defaultdict(list)
	preds = defaultdict(list)
	truths = defaultdict(list)
	outputs_raw = defaultdict(list)

	# Get all col names by default. The first 2 cols are image names
	if col_used is None:
		col_used = [test_frame.columns[i] for i in range(2, len(col_used))]

	if new_img_dir:
		test_frame = check_imgs(test_frame, new_img_dir)

	for index, row in tqdm(test_frame.iterrows()):
		image_name = row['face_name_align']
		    
		image = Image.open(image_name).convert('RGB')    
		image = trans(image)
		image_width = int((torch.prod(torch.tensor(image.shape)) / 3) ** 0.5)
		image = image.view(1, 3, image_width, image_width)
		image = image.to(device)
		output_i = model(image)
		output_i = output_i.cpu().detach().numpy()
		outputs = np.squeeze(output_i)

		for col, st_end in zip(col_used, output_cols_each_task):
			outputs_this_col = outputs[st_end[0]:st_end[1]]
			outputs_raw[col].append(outputs_this_col)

			score_this_col = np.exp(outputs_this_col) / np.sum(np.exp(outputs_this_col))
			scores[col].append(score_this_col)

			preds_this_cols = np.argmax(score_this_col)
			preds[col].append(preds_this_cols)

			truths[col].append(row[col])
		
		face_names.append(image_name)

	fair_df_cols, fair_df_col_names = [face_names, list(test_frame[sensitive_group])], ['face_name_align', sensitive_group]

	fair_df_cols += [scores[col] for col in col_used]
	fair_df_cols += [outputs_raw[col] for col in col_used]
	fair_df_cols += [preds[col] for col in col_used]
	fair_df_cols += [truths[col] for col in col_used]

	fair_df_col_names += [col + "_scores_fair" for col in col_used]
	fair_df_col_names += [col + "_outputs" for col in col_used]
	fair_df_col_names += [col + "_preds_fair" for col in col_used]
	fair_df_col_names += [col for col in col_used]

	fair_test = pd.DataFrame(fair_df_cols).T
	fair_test.columns = fair_df_col_names

	return fair_test

def print_acc_scores(fair_df, col_used=['race'], sensitive_group="gender"):
    unq_groups = set(fair_df[sensitive_group])
    acc_each_task = [0] * len(col_used)
    acc_each_group = [[0] * len(col_used) for g in unq_groups]
    n_row = fair_df.shape[0]
    for i,col in enumerate(col_used):
        acc_each_task[i] = sum(fair_df[col] == fair_df[col+"_preds_fair"]) / n_row
        for g in unq_groups:
            idx_g = fair_df[sensitive_group] == g
            n_g = sum(idx_g)
            acc_each_group[g][i] = sum(fair_df[col][idx_g ] == fair_df[col+"_preds_fair"][idx_g]) / n_g

    acc_overall = sum(acc_each_task) / n_row
    acc_each_group = [sum(acc_each_group_task) / len(acc_each_group_task) for acc_each_group_task in acc_each_group]
    
    return {"acc":acc_overall, "acc_each_group":acc_each_group}

def show_acc_df(fair_df, fair_df_full, col_used, sensitive_group):
    acc_scores = print_acc_scores(fair_df, col_used, sensitive_group)
    acc_scores_full = print_acc_scores(full_fair_df, col_used, sensitive_group)
    acc_scores = [acc_scores['acc']] + acc_scores['acc_each_group']
    acc_scores_full = [acc_scores_full['acc']] + acc_scores_full['acc_each_group']
    acc_df = {"Type":['No pruning', prune_type, "Difference"]}
    for i in range(len(acc_scores)):
        acc_df[i] = [acc_scores[i], acc_scores_full[i], acc_scores[i] - acc_scores_full[i]]
    acc_df = pd.DataFrame(acc_df)
    acc_df.columns = ['Type', "Overall"] + ['Group ' + str(i) for i in range(len(acc_scores))]
    print("Accuracy:")
    print(acc_df)
    print("Std(difference)".format(stdev(acc_df.loc[3,2:])))

def save_model(best_model, cfgs):
    # save final model
    prune_type,dataset,prune_type,loss_type,sensitive_group,network,prune_rate, exp_idx = cfgs
    model_path = 'trained_model/{}/{}_{}_{}_by{}_{}_{}_{}.pt'.format(prune_type,dataset,prune_type,loss_type,sensitive_group,network,prune_rate, exp_idx)
    while model_path in os.listdir('trained_model/{}'.format(prune_type)):
        exp_idx += 1
        model_path = 'trained_model/{}/{}_{}_{}_by{}_{}_{}_{}.pt'.format(prune_type,dataset,prune_type,loss_type,sensitive_group,network,prune_rate, exp_idx)            
    torch.save(best_model.state_dict(), model_path)
    print("Model saved at: {}".format(model_path))

def save_output(best_model, cfgs, csv_savedir="fair_dfs", save_file=True):
    dataset, prune_type, loss_type, prune_rate, test_frame, face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group,exp_idx = cfgs
    fair_df = check_fairness0(best_model, test_frame, face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group)
    df_path = '{}/{}_{}_{}_by{}_{}_{}_{}.csv'.format(csv_savedir,dataset,prune_type,loss_type,sensitive_group,network,prune_rate,exp_idx)
    while df_path in os.listdir(csv_savedir):
        exp_idx += 1
        df_path = '{}/{}_{}_{}_by{}_{}_{}_{}.csv'.format(csv_savedir,dataset,prune_type,loss_type,sensitive_group,network,prune_rate,exp_idx)
    if save_file:
        fair_df.to_csv(df_path)
        print("Output saved at: {}".format(df_path))
    return fair_df

############
# from SNIP, forward functions that pass through masks
############
def custom_forward_conv2d(self, x):
    #print("Passing through conv2d, ", self.mask.shape)
    return F.conv2d(x, self.weight * self.mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def custom_forward_conv1d(self, x):
    return F.conv1d(x, self.weight * self.mask, self.bias)

def custom_forward_linear(self, x):
    return F.linear(x, self.weight * self.mask, self.bias)

#############
# Automatically download datasets and prepare for loading
# Request Imagenet person subtree through database website.
#############
def download_dataset(dataset,img_dir):
    # Check if dataset has been save in imgdir already. If so returns
    if os.path.isdir(img_dir):
        print("Dataset exists!")
        return
    elif dataset == "Imagenet":
        raise FileNotFoundError("Imagenet not found. Request the person subtree through the database website, and save all images under Images/Imagenet")

    urls = {"FairFace":"https://drive.google.com/u/0/uc?id=1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86&export=download",
            "UTKFace":"https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=sharing&resourcekey=0-dabpv_3J0C0cditpiAfhAw",
            "CelebA": "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ"}
    url = urls[dataset]

    # Code from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    output = 'Images/{}.zip'.format(dataset)
    gdown.download(url, output, quiet=False, fuzzy=True)

    # extract zip file
    if dataset == 'UTKFace':
        file = tarfile.open(output)
        file.extractall(img_dir)
        file.close()
    else:
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(img_dir)

    # Move images to the same folder, then remove all subfolders.
    subfolders = [x[0] for x in os.walk(img_dir)][1:]
    for subfolder in subfolders:
        files = os.listdir(subfolder)
        for file in files:
            src = os.path.join(subfolder, file)
            # FairFace images comes with train/val split. But we reshuffle them all together to add a test split. 
            # So FairFace image names will have prefix "train" or "val".
            if dataset == 'FairFace':
                split = subfolder.split("/")[-1]
                dst = os.path.join(img_dir, "_".join([split,file]))
            else:
                dst = os.path.join(img_dir, file)
            shutil.move(src, dst)
        os.rmdir(subfolder)

    print("Dataset downloaded!")
    
    
def setseed(seed):
    # Set random seeds for training/eval
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_impt(cfgs):
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
        
