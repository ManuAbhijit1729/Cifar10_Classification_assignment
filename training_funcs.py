import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torchvision
import albumentations as A
import torch.nn.functional as F
import evaluation_metrics
import time
from utils import AverageMeter
from torch_ema import ExponentialMovingAverage
import subprocess as sp
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def val_model(model, device, val_loader, criterion, num_classes,epoch):
    model.eval()
    total_loss = 0
    val_iters = 0
    all_predictions = [] #update
    all_targets = [] #update
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start_epoch_time = time.time()
    conf_mat = conf_mat = np.zeros((num_classes,num_classes))
    print("validation...")
    with torch.no_grad():
        progress_bar = tqdm(val_loader)
        for batch_idx,(img, label) in enumerate(progress_bar):
            data_time.update(time.time() - start_epoch_time)
            img, label = img.to(device = device), label.to(device = device)
            output = model(img)
            val_loss = criterion(output,label.long()) # Computing loss
            # val_loss = criterion(output,torch.squeeze(label,dim = 1).long())
            total_loss += val_loss
            preds = torch.argmax(torch.nn.Softmax(dim = 1)(output),dim = 1)
            all_predictions.extend(preds.cpu().numpy()) #update
            all_targets.extend(label.squeeze().cpu().numpy()) # update
            conf_mat += evaluation_metrics.create_confusion_matrix(label = label, output = preds, num_classes = num_classes)
            val_iters += 1
            batch_time.update(time.time() - start_epoch_time)
            losses.update(val_loss.item(), img.shape[0])
            start_epoch_time = time.time()
            progress_bar.set_description('(Epoch {epoch} | {batch}/{size}) | Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}| GPU usage: {gpu_use:.3f} GB'.format(
                    epoch = epoch + 1,
                    batch=batch_idx + 1,
                    size = len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    gpu_use = get_memory_usage(device)
                    ))
    
    all_predictions = np.array(all_predictions) #update
    all_targets = np.array(all_targets) #update
    # Precision, Recall, F1-Score
    precision = precision_score(all_targets, all_predictions, average='macro')  # macro precision using sklearn
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # mean_precision, mean_recall, mean_accuracy, mean_F1 = evaluation_metrics.calculate(conf_mat,num_classes) # To compute metrics using my custom evaluation_metrics module
    loss = total_loss/val_iters

    return loss, precision, recall, accuracy, f1     

# Saving model having minimum absolute difference between train and val loss
def model_saver(model, scaler, optimizer, scheduler, training_des, lr, model_name, save_dir, val_loss, train_loss, min_loss_mean, min_loss_diff, epoch):
    loss_diff = abs(val_loss - train_loss)
    loss_mean = (val_loss + train_loss)/2
    if epoch < 5:
        last_path = 'Last_Model_{}_{}_{}.pt'.format(training_des,lr,model_name)
        torch.save({'model_state_dict' : model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'scalar_state': scaler.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch':epoch,
                'min_loss_diff': min_loss_diff,
                'min_loss_mean': min_loss_mean}, os.path.join(save_dir, last_path))
        return min_loss_mean, min_loss_diff
    if loss_diff < min_loss_diff or loss_mean < min_loss_mean:
        print('saving with validation loss of {}'.format(val_loss))
        min_loss_diff = loss_diff
        min_loss_mean = loss_mean
        best_path = 'Best_Model_{}_{}_{}_{}.pt'.format(training_des,lr,model_name,epoch)
        torch.save({'model_state_dict' : model.state_dict(),}, os.path.join(save_dir,best_path))
    last_path = 'Last_Model_{}_{}_{}_{}.pt'.format(training_des,lr,model_name,epoch)
    # last_path = 'Last_Model_{}_{}_{}.pt'.format(training_des,lr,model_name)
    torch.save({'model_state_dict' : model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'scalar_state': scaler.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch':epoch,
                'min_loss_diff': min_loss_diff,
                'min_loss_mean': min_loss_mean}, os.path.join(save_dir, last_path))
    return min_loss_mean, min_loss_diff


def get_memory_usage(device): #To compute memory usage, in GB
    device = device.index
    command = f'nvidia-smi --query-gpu=memory.used --id={device} --format=csv'
    out = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    used_memory = float(out[0].split()[0])
    return used_memory/1e3

def train_model(model, train_loader, num_epochs, writer_train, writer_val, device, val_loader, optimizer, criterion, num_classes, scheduler, training_des, lr, model_name, save_dir,grad_accum, start_epoch, min_loss_diff, min_loss_mean, scaler):
    min_loss_mean = min_loss_mean
    min_loss_diff = min_loss_diff
    
    ema = ExponentialMovingAverage(model.parameters(), decay = 0.998) # Using torch_ema to improve model performance
    conf_mat = np.zeros((num_classes,num_classes))
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        model.train()
        train_iters = 0
        all_predictions = [] #update
        all_targets = [] #update
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        start_epoch_time = time.time()
        print(f'[Epoch: {epoch + 1} training...]')
        progress_bar = tqdm(train_loader) # To display progress stats
        for batch_idx,(input, label) in enumerate(progress_bar):
            data_time.update(time.time() - start_epoch_time)
            input, label = input.to(device = device), label.to(device = device) # Putiing data onto GPU memory
    
            with torch.autocast(device_type = 'cuda', dtype = torch.float16): #mixed precision training . Runs the forward pass with autocasting.
                output = model(input)
                # print(output.shape)
                # print(label.shape)
                loss = criterion(output,label.long()) / grad_accum
                # loss = criterion(output,torch.squeeze(label,dim = 1).long()) / grad_accum

            scaler.scale(loss).backward() # Scales loss.  Calls backward() on scaled loss to create scaled gradients.

            if ((batch_idx + 1) % grad_accum == 0) or (batch_idx + 1 == len(train_loader)): # Gradient Accumulation updates gradients every grad_accum steps
                scaler.unscale_(optimizer) #scaler.unscale_() first unscales the gradients of the optimizer's assigned params.  
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clipping to ensure we do not encounter infs or NaNs.
                scaler.step(optimizer) # scaler.step() - # optimizer's gradients are already unscaled, so scaler.step does not unscale them, If these gradients do not contain infs or NaNs, optimizer.step() is then called.
                scaler.update() # Updates the scale for next iteration.
                optimizer.zero_grad() # zero the parameter gradients
                ema.update() # update the ema parameters after Backward pass.
                
            preds = torch.argmax(torch.nn.Softmax(dim = 1)(output),dim = 1) #Final predictions
            
            all_predictions.extend(preds.cpu().numpy()) #update
            all_targets.extend(label.squeeze().cpu().numpy()) # update
        
            conf_mat += evaluation_metrics.create_confusion_matrix(label = label, output = preds, num_classes = num_classes)
            running_loss += loss.item()
            train_iters += 1
            batch_time.update(time.time() - start_epoch_time)
            losses.update(loss.item(),input.shape[0])
            start_epoch_time = time.time()
            progress_bar.set_description('(Epoch {epoch} | {batch}/{size}) | Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | GPU usage: {gpu_use:.3f} GB'.format(
                    epoch = epoch + 1,
                    batch=batch_idx + 1,
                    size = len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    gpu_use = get_memory_usage(device)
                    # gpu_use = 0
                    ))
            
        all_predictions = np.array(all_predictions) #update
        all_targets = np.array(all_targets) #update
        
        # Precision, Recall, F1-Score
        precision = precision_score(all_targets, all_predictions, average='macro')  # macro precision using sklearn
        recall = recall_score(all_targets, all_predictions, average='macro')
        f1 = f1_score(all_targets, all_predictions, average='macro')
        accuracy = accuracy_score(all_targets, all_predictions)
        # disp = ConfusionMatrixDisplay.from_predictions(conf_mat,np.array(range(10))) # displaying confusion matrix
     
        
        # mean_precision, mean_recall, mean_accuracy, mean_F1 = evaluation_metrics.calculate(conf_mat,num_classes) # To compute metrics my custom evaluation_metrics module
        train_loss = running_loss/train_iters
        scheduler.step(epoch)
        print(f'learning rate is {optimizer.param_groups[0]["lr"]}')
        print(f'training loss: {train_loss :.5f}')
        print(f'training accuracy: {accuracy :.5f}')
        print(f'training precision: {precision :.5f}')
        print(f'training recall: {recall :.5f}')
        print(f'training F1-Score: {f1 :.5f}')
        
        ConfusionMatrixDisplay.from_predictions(all_targets,all_predictions) # displaying confusion matrix
        plt.show()
        
        writer_train.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step = epoch)
        writer_train.add_scalar('Loss', train_loss, global_step = epoch)
        writer_train.add_scalar('Accuracy',accuracy, global_step = epoch)
        writer_train.add_scalar('Precision',precision, global_step = epoch)
        writer_train.add_scalar('Recall',recall, global_step = epoch)
        writer_train.add_scalar('F1-Score',f1, global_step = epoch)

        #Model Validation
        with ema.average_parameters():
            val_loss, val_precision, val_recall, val_accuracy, val_f1 = val_model(model, device, val_loader, criterion, num_classes, epoch)

        print(f'validation loss: {val_loss :.5f}')
        print(f'validation accuracy: {val_precision :.5f}')
        print(f'validation precision: {val_recall :.5f}')
        print(f'validation recall: {val_accuracy :.5f}')
        print(f'validation F1-Score: {val_f1 :.5f}')
        
        writer_val.add_scalar("Loss", val_loss, global_step= epoch)
        writer_val.add_scalar('Accuracy',val_accuracy, global_step = epoch)
        writer_val.add_scalar('Precision',val_precision, global_step = epoch)
        writer_val.add_scalar('Recall',val_recall, global_step = epoch)
        writer_val.add_scalar('F1-Score',val_f1, global_step = epoch)
        with ema.average_parameters(): # saving model using ema parameters
            min_loss_mean, min_loss_diff = model_saver(model, scaler, optimizer, scheduler, training_des, lr, model_name, save_dir, val_loss, train_loss, min_loss_mean, min_loss_diff, epoch)
        

def test_model(model, device, test_loader):
    model.eval()
    all_predictions = [] #update
    all_targets = [] #update
    batch_time = AverageMeter()
    data_time = AverageMeter()
    start_epoch_time = time.time()
    print("Testing...")
    with torch.no_grad():
        progress_bar = tqdm(test_loader)
        for batch_idx,(img, label) in enumerate(progress_bar):
            data_time.update(time.time() - start_epoch_time)
            img, label = img.to(device = device), label.to(device = device)
            output = model(img)
            preds = torch.argmax(torch.nn.Softmax(dim = 1)(output),dim = 1)
            all_predictions.extend(preds.cpu().numpy()) #update
            all_targets.extend(label.squeeze().cpu().numpy()) # update
            batch_time.update(time.time() - start_epoch_time)
            start_epoch_time = time.time()
            progress_bar.set_description('({batch}/{size}) | Data: {data:.3f}s | Batch: {bt:.3f}s | GPU usage: {gpu_use:.3f} GB'.format(
                    batch=batch_idx + 1,
                    size = len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    gpu_use = get_memory_usage(device)
                    ))
    
    all_predictions = np.array(all_predictions) #update
    all_targets = np.array(all_targets) #update
    # Precision, Recall, F1-Score
    precision = precision_score(all_targets, all_predictions, average='macro')  # macro precision using sklearn
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    accuracy = accuracy_score(all_targets, all_predictions)
    
    print(f'testing accuracy: {accuracy :.5f}')
    print(f'testing precision: {precision :.5f}')
    print(f'testing recall: {recall :.5f}')
    print(f'testing F1-Score: {f1 :.5f}')
    
    ConfusionMatrixDisplay.from_predictions(all_targets,all_predictions) # displaying confusion matrix
    plt.show()
    
    
      

#Getting predictions on individual images
def final_preds(model, img, device):
    transforms_val = A.Compose([A.Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))])
    model.eval()
    with torch.no_grad():
        img = transforms_val(image = img)['image']
        img = torchvision.transforms.functional.to_tensor(img)
        img = torch.unsqueeze(img,0)
        img = img.to(device = device)
        output = model(img)
        pred = torch.argmax(F.softmax(output,dim=1),axis=1).cpu().numpy()
        pred = pred.astype(np.int16)
    return pred