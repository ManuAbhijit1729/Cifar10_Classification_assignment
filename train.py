import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import timm
import timm.data
import timm.loss
import timm.optim
import timm.utils
from timm.scheduler import CosineLRScheduler
import albumentations as A
import torchvision.datasets as datasets
import utils
import training_funcs
from albumentations.pytorch.transforms import ToTensorV2 
import torchvision.transforms as transforms
from pathlib import Path
import torchvision.datasets as datasets
from custom_model import ConvNet

if __name__ == '__main__':    
    
    utils.seed_everything(420)  # seed random number generator (To make everything deterministic/reproducible)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)
    
        #Transforms
    transform_train= transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
    ])

    #Data Preparation

    #Basic setup


    root=r"C:\Users\satya\Downloads\Cifar10"    #Change root and base_path and run the script
    base_path = Path(r"C:\Users\satya\Downloads\Runs") # Path where the tensorboard event files and training weights are saved


    cifar10_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    # Extract digit classes (0-9)
    digit_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_ratio = 0.8
    # Create subsets for digits
    train_indices = [i for i, (_, label) in enumerate(cifar10_train) if label in digit_classes]
    train_labels = [cifar10_train.targets[i] for i in train_indices]

    # Perform stratified split into train and validation sets
    train_indices, val_indices = train_test_split(
        train_indices,
        train_size=train_ratio,
        stratify=train_labels,
        random_state=42
    )

    print(f'Train data: {len(train_indices)}, Val data: {len(val_indices)}')


    #Model selection and parameters

    model_name = 'resnet_50'
    num_classes = 10

    model = timm.create_model('resnet50', num_classes=num_classes, pretrained=True)
    # model = ConvNet()

    model=model.to(device)
    for params in model.parameters():
        params.requires_grad = True

    print(f'Model Name : ',model_name)
    print(f"Total number of trainable parameters in model {sum(p.numel() for p in model.parameters() if p.requires_grad is True)}")

    #Hyperparameters

    batch_size = 128
    # lr = 0.0001
    lr = 4e-4
    num_epochs = 60
    grad_accum = 1

    #Loss Function
    criterion = nn.CrossEntropyLoss().to(device=device)
    #Optimizer
    optimizer = timm.optim.AdamW(model.parameters(), lr = lr)
    #Scheduler
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial = num_epochs, lr_min = 1e-6)
    #scaler
    scaler = torch.amp.GradScaler()

    #Data and paths set up
    train_subset = Subset(cifar10_train, train_indices)
    val_subset = Subset(cifar10_train, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    iteration='1'

    #Model details and tensorboard
    training_des = 'with_overlay'

    tb_path = f'Iteration_{iteration}_{model_name}'

    tensorboard_log_dir = base_path / 'tensorboard_logs'
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)  # Creates the directory if it doesn't exist

    # Create TensorBoard writers
    writer_train = SummaryWriter(tensorboard_log_dir / f'{tb_path}_train')
    writer_val = SummaryWriter(tensorboard_log_dir / f'{tb_path}_val')

    if not os.path.exists(os.path.join(base_path, 'tensorboard_logs')):
        os.mkdir(os.path.join(base_path, 'tensorboard_logs'))

    #Model saving path
    save_dir = os.path.join(base_path,'training_weights')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    resume = False # Setting resume variable true to resume training from last epoch trained
    resume_checkpoint = None
    ##Ensure the base_path for checkpoint. This is to ensure the tensorboard logs are correct
    if resume:
        assert os.path.isfile(resume_checkpoint), 'The path does not exist'
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        scaler.load_state_dict(checkpoint['scalar_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss_diff = checkpoint['min_loss_diff']
        min_loss_mean = checkpoint['min_loss_mean']

    else:
        start_epoch = 0
        min_loss_diff = 1000
        min_loss_mean = 1000
            

    training_funcs.train_model(model, train_loader, num_epochs, writer_train, writer_val, device, val_loader, optimizer, criterion, num_classes, scheduler, training_des, lr, model_name, save_dir, grad_accum, start_epoch, min_loss_diff, min_loss_mean, scaler)

