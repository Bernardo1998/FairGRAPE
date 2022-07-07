import torch
import torch.optim as optim
import datetime
from torchvision import models, transforms
import copy
import pandas as pd
import numpy as np

#############
# Below are training sections
#############

def train(best_model, criterion, dataloaders, learning_rates = [1e-4,1e-5,1e-6], epoches = [13,3,3], col_used_training=[0,1,2], output_cols_each_task=[(0,7),(7,9),(9,18)]):
    best_acc, best_loss = 0, 100
    for learning_rate, epoch in zip(learning_rates, epoches):
        print('learning rate:', learning_rate)
        print('epoch number:', epoch)
        print('Current time:', datetime.datetime.now())
        optimizer_conv = optim.Adam(best_model.parameters(), lr=learning_rate)
        current_model, current_acc, current_loss = train_model0(best_model, dataloaders, criterion,optimizer_conv,epoch,col_used_training,output_cols_each_task)

        if current_loss < best_loss:
        #if current_acc > best_acc:
            best_acc = current_acc
            best_loss = current_loss
            best_model = current_model
            print('best acc, loss update:', best_acc, best_loss)
        else:
            print('best acc, loss not update, still:', best_acc, best_loss)

    torch.cuda.empty_cache()
    return best_model

def train_model0(model, dataloaders, criterion, optimizer, num_epochs=25, col_used_training=[0,1,2], output_cols_each_task=[(0,7),(7,9),(9,18)], detail_acc = True, show_epoch=200):
    device = torch.device('cuda:0')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100

    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            i = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # running = current epoch    
            running_loss = 0.0
            # Following is count of batch that are correct. Max is 64 (batch size)
            running_corrects = [0] * len(col_used_training)
    
            # Iterate over data.
            # Get either train or test dataset
            for sample_batched in dataloaders[phase]:
                
                image_batched, label_batched = sample_batched
                image_batched = image_batched.to(device, dtype=torch.float)
                # transfer it all to gpu
                # Exclude the last label since it will be the sensitive group.
                label_batched = label_batched[:, 0:len(col_used_training)].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history (we want to calculate the gradient via set_grad_enabled) if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Evaluate model on image, then squeeze (like "unlist" in R)
                    outputs = torch.squeeze(model(image_batched))
                    loss,acc = loss_multi_tasks(outputs, label_batched, criterion, output_cols_each_task,True)                    
                    i = i + 1   
                    
                    
                    if i % show_epoch == 0:
                        print('batch:', i)
                        print('batch loss:', loss.item())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_corrects = [running_corrects[i] + acc[i] for i in range(len(col_used_training))]
                running_loss += loss.item() * image_batched.size(0) # average loss * batch size = running loss          
                # Batch Ends here
            
            # Average losses for each epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = sum(running_corrects).double() / (len(dataloaders[phase].dataset) * len(col_used_training))

            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if detail_acc:
                acc_details = ""
                for i in range(len(col_used_training)):
                    acc_details += "{} acc: {}. ".format(col_used_training[i], running_corrects[i]/len(dataloaders[phase].dataset))
                print(acc_details)

            if phase == 'test' and epoch_loss < best_loss:
                # We started with ImageNet model with 100 los and 0 accuracy
                # If we get a better epoch loss and accuracy, update model
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    print("Training complete, best acc: ", best_acc, type(best_acc))
    return model, best_acc, best_loss

def loss_multi_tasks(outputs, labels, criterion=None, output_cols_each_task=[(0,7),(7,9),(9,18)], find_acc=False):
    tasks_outputs, tasks_labels, ntasks = [], [], len(output_cols_each_task)
    for i, st_ed in enumerate(output_cols_each_task):
        #print(i, st_ed, outputs.shape, labels.shape)
        tasks_outputs.append(outputs[:, st_ed[0]:st_ed[1]])
        tasks_labels.append(labels[:, i])
                 
    if criterion:
        loss = 0
        for i in range(ntasks):
            loss += criterion(tasks_outputs[i], tasks_labels[i])
        loss = loss.double() / ntasks 
    if find_acc:
        acc = [0] * ntasks
        for i in range(ntasks):
            _, task_preds = torch.max(tasks_outputs[i], 1)
            acc[i] = torch.sum(task_preds.cpu() == tasks_labels[i].cpu())

    if find_acc:
        return loss, acc
    else:
        return loss

