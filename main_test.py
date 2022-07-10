import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import argparse
import datetime

from dataset import make_frame, make_datasets
from prune import WS, SNIP, GraSP, Lottery, FairGRAPE, Importance, Random
from util import make_model, save_model, save_output, download_dataset, show_acc_df
from train_and_val import train

pruner_map = {'WS':WS, "SNIP":SNIP,'GraSP':GraSP,"Lottery":Lottery,"FairGRAPE":FairGRAPE, "Importance":Importance, "Random":Random}

def experiment(args):
    checkpoint = args.checkpoint # previously pruned models
    dataset = args.dataset # ['UTKFace', 'FairFace', "CelebA", "ImbalancedFairFace"]
    network = args.network # ['resnet34', 'mobilenetv2']
    loss_type = args.loss_type # ['race', 'raceAndgender', 'gender', 'attrs', 'class']
    sensitive_group = args.sensitive_group # ['race', 'raceAndgender', 'gender']
    prune_type = args.prune_type # ['FairGRAPE','SNIP','WS','Lottery', 'GraSP','Full',"readResult"]
    prune_rate = args.prune_rate # [0.5, 0.7, 0.8, 0.9, 0.99]
    batch_size = args.batch
    init_train = args.init_train == 1
    drop_race = args.drop_race # See util.make_frame() for detail
    retrain = args.retrain == 1
    save_mask = args.save_mask == 1
    delta_p = args.delta_p
    print_acc = args.print_acc == 1
    exp_idx = args.exp_idx 


    # Set random seeds for training/eval
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make dir for saving results
    save_dir = "trained_model/{}".format(prune_type)
    csv_savedir = "fair_dfs"
    dirs = [csv_savedir, 'models', save_dir,"Images"]
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)
    
    print("Type:{}, Network:{}, Sparsity:{}, Dataset:{}".format(prune_type, network, prune_rate,dataset))

    if dataset == 'FairFace':
        csv = 'csv/FairFace.csv'
        face_dir = 'Images/FairFace'
        download_dataset(dataset, face_dir)
        # Which variables are used in training.          
        if loss_type == 'race':
            total_classes, output_cols_each_task,col_used_training = 7, [(0,7)], [loss_type]
        elif loss_type == 'gender':
            total_classes, output_cols_each_task,col_used_training = 2, [(0,2)], [loss_type]
        else:
            total_classes, output_cols_each_task,col_used_training = 9, [(0,7),(7,9)], ['race','gender']
        # col_used includes a sensitive group label. It will be used for FairGRAPE pruning, but not in training stage.
        # When making the dataset we used col_used to that the sensitive group is included
        # when trainging the model for a given task we exclude sensitive group information
        col_used = col_used_training + [sensitive_group]
        epoches = [13,3,3]
        frames = make_frame(csv, face_dir)
        if drop_race:
            frames_minority = make_frame(csv, face_dir, drop_race=drop_race)
            train_loader_minority,_ = make_datasets(frames_minority['train'], frames_minority['val'], True, batch_size,col_used)
    elif dataset == 'UTKFace':
        csv = 'csv/UTKFace_labels.csv'
        face_dir = 'Images/UTKFace'
        download_dataset(dataset, face_dir)
        # Which variables are used in training.          
        if loss_type == 'race':
            total_classes, output_cols_each_task,col_used_training = 4, [(0,4)], [loss_type]
        elif loss_type == 'gender':
            total_classes, output_cols_each_task,col_used_training = 2, [(0,2)], [loss_type]
        else:
            total_classes, output_cols_each_task,col_used_training = 6, [(0,4),(4,6)], ['race','gender']
        col_used = col_used_training + [sensitive_group]
        epoches = [13,3,3]
        frames = make_frame(csv, face_dir, seven_races=False)
        if drop_race:
            frames_minority = make_frame(csv, face_dir, seven_races=False, drop_race=drop_race)
            train_loader_minority,_ = make_datasets(frames_minority['train'], frames_minority['val'], True, batch_size,col_used)
    elif dataset == "CelebA":
        csv = 'csv/CelebA.csv'
        face_dir = 'Images/CelebA/img_align_celeba'
        download_dataset(dataset, face_dir)
        frames = make_frame(csv, face_dir, seven_races=False)
        # Which variables are used in training. 
        output_cols_each_task = [(i*2, i*2+2)  for i in range(39) ] 
        col_used_training = [frames['train'].columns[i] for i in range(2, 41)]
        total_classes = 39*2 # Gender removed from training. "Classes" here actually mean columns in the output
        col_used = col_used_training + [sensitive_group]
        epoches = [8,1,1]
    elif dataset == "Imagenet":
        csv = 'csv/Imagenet.csv'
        face_dir = 'Images/Imagenet'
        download_dataset(dataset, face_dir)
        frames = make_frame(csv, face_dir, seven_races=False)
        # Which variables are used in training. 
        output_cols_each_task = [(0,104)]
        col_used_training = ['classes'] 
        total_classes = 104 # Gender removed from training 
        col_used = col_used_training + [sensitive_group]
        epoches = [8,5,5]      
    else:
        raise NotImplementedError("{} is not implemented!".format(dataset))

    #print("Col_used:",col_used)
    lr_schedule = [1e-4, 1e-5,1e-6]
    train_loader, test_loader = make_datasets(frames['train'], frames['val'], True, batch_size,col_used)
    dataloaders = {'train':train_loader, 'test':test_loader}
    
    save_model_iter = args.save_model_iter
    print(save_model_iter)

    device = torch.device('cuda:0')
    criterion = nn.CrossEntropyLoss()

    torch.cuda.empty_cache()
    best_model = make_model(network=network,n_classes=total_classes).to(device)

    ########################
    # Set parameters needed for each pruning
    ########################
    if prune_type == 'WS':
        prune_cfgs = [prune_rate]
    elif prune_type == 'SNIP':
        num_batch_sampling, var_scaling = 1, True
        prune_cfgs = [prune_rate, num_batch_sampling, var_scaling]
    elif prune_type == 'Lottery':
        prune_cfgs = [prune_rate]
    elif prune_type == 'GraSP':
        # GraSP selects a batch balanced w.r.t output classes (not sensitive groups) for signal calculation.
	# In CelebA and Imagenet, the numbers of classes are large, limiting the samples_per_class.
        if len(col_used_training) > 1:
            target_col, samples_per_class, num_classes = [i for i in range(len(col_used_training))], 1, len(col_used_training)
        else:
            target_col, samples_per_class, num_classes = len(col_used_training) - 1 , 10, total_classes
        if drop_race and loss_type == 'race':
            num_classes = num_classes - 1 if drop_race < 10 else 1
        prune_cfgs = [prune_rate, target_col, num_classes, samples_per_class]
    elif prune_type == "FairGRAPE" or prune_type == "Importance":
        sensitive_classes = len(set(frames['train'][sensitive_group]))
        masked_grads = True
        impt = args.impt #[0, 1, 2]
        if impt == 2:
            sensitive_classes = total_classes
        para_batch = args.para_batch
        stop_batch = args.stop_batch
        prune_cfgs = [prune_rate, frames['train'], face_dir, sensitive_classes, masked_grads, output_cols_each_task ,col_used, para_batch, impt, stop_batch, delta_p]
    elif prune_type == "Random":
        prune_cfgs = [prune_rate, True]
    elif prune_type == 'Full':
        prune_rate = 0
    else:
        raise NotImplementedError("Prune method {} is not implemented!".format(prune_type))

    ########################
    # Set iterative pruning. If prune_iter = 1 then it`s single shot
    #########################
    pct_remain_after_this_iter = 1 - args.init_pruned
    if prune_type in ['WS', 'SNIP', 'GraSP','Full','Random']:
        prune_iters = 1 if prune_type != 'Full' else 0
        retrain_lr = 0
        retrain_iters = 0
        keep_per_iter = 1-prune_rate
        lr_decay_iter = 1
    elif prune_type in ['FairGRAPE','Lottery', 'Importance']:
        prune_iters = args.prune_iter
        retrain_lr = args.retrain_lr
        retrain_iters = args.retrain_iter
        keep_per_iter = args.keep_per_iter 
        lr_decay_iter = args.lr_decay_iter
        # determine parameters are needed if not specified
        prune_iters = math.ceil(math.log((1-prune_rate)/pct_remain_after_this_iter, keep_per_iter)) if prune_iters is None else prune_iters 
        lr_decay_iter = int(prune_iters * 0.7) if lr_decay_iter is None else lr_decay_iter 

    if init_train and prune_type in ['WS', 'Full', 'FairGRAPE','Lottery', 'Importance'] or print_acc:
        print("Training before pruning!" if prune_type != 'Full' else "No pruning, full training!")
        best_model = train(best_model, criterion, dataloaders,lr_schedule, epoches,col_used_training, output_cols_each_task) 
        full_fair_df = save_output(best_model,[dataset, prune_type,loss_type,prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx],csv_savedir, False)

    print("Iters to prune:", prune_iters)
    if prune_type in pruner_map:
        prune_loader = train_loader if not drop_race else train_loader_minority
        prunner = pruner_map[prune_type](best_model, criterion, prune_loader,output_cols_each_task, save_mask)

    if checkpoint is not None:
        print("Loading checkpoint from {}".format(checkpoint))
        # Checkpoints contain mask attributes in layers. Must init before loading.
        prunner.init_mask() 
        best_model = prunner.get_model()
        best_model.load_state_dict(torch.load(checkpoint))
        best_model = best_model.to(device)
          
    # Main pruning iter
    for i in range(prune_iters):
        print('Current time:', datetime.datetime.now())
        pct_remain_after_this_iter *= keep_per_iter 
        # Make sure pruned parameters do not go beyond desired sparsity rate.
        accumulated_pruned = min(1-pct_remain_after_this_iter, prune_rate)
        print("\nPrune iter:{}, prop of weights remain after this iter:{}".format(i, 1-accumulated_pruned))
        prune_cfgs[0] = accumulated_pruned # Update the actual amount to keep for each iteration.
        best_model = prunner.prune(prune_cfgs, True)
        if retrain_iters > 0:
            best_model = train(best_model, criterion, dataloaders,[retrain_lr], [retrain_iters],col_used_training, output_cols_each_task)
        prunner.update_model(best_model) 
        # Save model at some iterations.
        if i in save_model_iter:
            save_model(best_model,[prune_type,dataset,prune_type,loss_type,sensitive_group,network,accumulated_pruned, exp_idx])


    # Retraining after pruning
    if prune_iters > 0 and retrain:
        print("Training after pruning!")
        best_model = train(best_model, criterion, dataloaders,lr_schedule, epoches,col_used_training, output_cols_each_task)        
    # Save model
    save_model(best_model,[prune_type,dataset,prune_type,loss_type,sensitive_group,network,prune_rate, exp_idx])

    # Save prediction output
    fair_df = save_output(best_model,[dataset, prune_type,loss_type,prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx],csv_savedir)

    # Print acc scores, overall and by groups
    if print_acc:
        show_acc_df(fair_df, fair_df_full, col_used, sensitive_group)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for pruning experiements')
    parser.add_argument('--checkpoint',type=str, default=None, help='Path to a trained model.')
    parser.add_argument('--dataset',type=str, default='FairFace', help='Dataset of Training')
    parser.add_argument('--network',type=str, default='resnet34', help='Network of Training')
    parser.add_argument('--prune_type',type=str, default='FairGRAPE', help='Pruning method to test')
    parser.add_argument('--loss_type',type=str, default='gender', help='Classification Tasks')
    parser.add_argument('--sensitive_group',type=str, default='gender', help='Sensitive group to control gradient for')
    parser.add_argument('--init_pruned',type=float, default=0, help='How many parameters already pruned.')
    parser.add_argument('--prune_rate',type=float, default=0.9, help='Desired Sparsity level')
    parser.add_argument('--prune_iter',type=int, default=None, help='Iterations in iterative pruning')
    parser.add_argument('--retrain_iter',type=int, default=3, help='Number of retraining after each pruning')
    parser.add_argument('--retrain_lr',type=float, default=1e-5, help='Learning rate of retraining')
    parser.add_argument('--keep_per_iter',type=float, default=0.9, help='Pruning step')
    parser.add_argument('--lr_decay_iter',type=int, default=15, help='Iterations after which learning rate would decay.')
    parser.add_argument('--batch',type=int, default=64, help='Batch size in dataloaders')
    parser.add_argument('--impt',type=int, default=0, help='Type of importance score to be returned')
    parser.add_argument('--para_batch',type=int, default=1, help='Parameters selected before updating race group in greedy method')
    parser.add_argument('--stop_batch',type=int, default=10000, help='Mini-batches of images used in importance calculation')
    parser.add_argument('--exp_idx',type=int, default=0, help='Index of current experiment')
    parser.add_argument('--init_train',type=int, default=0,help='Whether initial training is conducted')
    parser.add_argument('--drop_race', type=int, default=0,help="Dropping selected race(s) or not")
    parser.add_argument('--retrain', type=int, default=1,help="Retraining after pruning or not")
    parser.add_argument('--save_mask', type=int, default=0,help="Save pruning masks as an npy file")
    parser.add_argument('--print_acc', type=int, default=0,help="Show test acc after pruning and fine tuning")
    parser.add_argument('--delta_p', type=int, default=0, help="FG selects next node by i(0) or p(1), or i*p(2)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--save_model_iter', nargs='+', help='Save current model at selected iterations', type=int,default=-1)

    args = parser.parse_args()

    experiment(args)
