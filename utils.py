import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
import random
from blurgenerator import lens_blur
from PIL import Image, ImageFile
from albumentations.pytorch.transforms import ToTensorV2 
import pickle
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split



class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def blur(img_hm,p=0.2):
    if random.random()<p:
        radius = random.choice([5,7,10])
        img_hm = lens_blur(img_hm, radius=radius, components=4, exposure_gamma=2)
    return img_hm

class Custom_data(Dataset):
    def __init__(self, images, labels, transform = False):
        self.images = images
        self.label = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        Image.MAX_IMAGE_PIXELS=None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        img, label = np.asarray(self.images[idx]), np.asarray(self.label[idx])
        # print(f'input shape',img.shape)
 
        plt.imshow(img)
        
        if self.transform:
            tf = self.transform(image=img)
            img = tf['image']
        
        return (img, label)

def train_val_split(root,train_ratio=0.8):
    data = []
    targets = []
    train_data = [file for file in os.listdir(root) if "data_batch" in file]
    # test_data = [file for file in os.listdir(root) if "test_batch" in file]
            

    for files in train_data:
        entry = extract(os.path.join(root,files))
        data.append(entry["data"])
        targets.extend(entry["labels"])
            
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    targets = np.asarray(targets)
    data = data.transpose((0, 2, 3, 1))

    # 80/20 Train/Val Split 

    X_train, X_val, y_train, y_val = train_test_split(data,targets,train_size=train_ratio,random_state=42,stratify=targets) 
    
    return X_train, X_val, y_train, y_val

def train_val_split_torch(root,transform_train,transform_val,train_ratio = 0.8):
    
    cifar10_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    # Extract digit classes (0-9)
    digit_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
    
    return cifar10_train, train_indices, val_indices

def extract(filename):
    with open(filename,"rb") as f:
        batch_data = pickle.load(f,encoding="latin1")
    return batch_data  
    
def load_meta(root):
    path = os.path.join(root,"batches.meta")
    with open(path,"rb") as infile:
        data = pickle.load(infile,encoding="latin1")
        classes = data["label_names"]
        classes_to_idx = {_class:i for i,_class in enumerate(classes)}

def seed_everything(seed: int):       
    #random.seed(seed)            
    np.random.seed(seed)      
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)        
    torch.backends.cudnn.deterministic = True        
    torch.backends.cudnn.benchmark = True


#Visualising the augmentations
def view_as_grids(data):
    plt.figure(figsize=(12,6))
    plt.imshow(data)
    plt.axis('off')
    plt.show()
    
