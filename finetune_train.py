import os
import sys
import torch
import monai
import random
import fnmatch
import logging
import torch.nn as nn
from monai.metrics import ROCAUCMetric
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.nets import ViTAutoEnc
from torch.utils.tensorboard import SummaryWriter
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    LoadImaged,
    Compose,
    Resized,
    Activations, 
    AsDiscrete,
    ScaleIntensityd
)

from Models import Classifier


os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Select the GPU, default 0

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    def get_data(path):
        #normal && tumor  meningioma && glioma  IDH-wt && IDH-mut
        data_0_dir = os.path.join(path,'') #Name of the class labeled 0, e.g.: normal
        data_1_dir = os.path.join(path,'') #Name of the class labeled 1, e.g.: tumor
        
        data_0=list()    
        for path,dirs,files in os.walk(data_0_dir):
            for f in fnmatch.filter(files,'*.nii.gz'):
                data_0.append(os.path.join(path,f))             
                 
        data_1=list()
        for path,dirs,files in os.walk(data_1_dir):
            for f in fnmatch.filter(files,'*.nii.gz'):
                data_1.append(os.path.join(path,f)) 
           
        # 2 binary labels for classification: tumor and normal
        files_0 = [{"img": img[0], "label": 0} for img in zip(data_0)]
        files_1 = [{"img": img[0], "label": 1} for img in zip(data_1)]
        data = files_0 + files_1
        random.shuffle(data)
        print('Total Number of training Data Samples:{}'.format(len(data)))
        return(data)
    
    train_files = get_data('/path/to/train_data')
    val_files = get_data('/path/to/validation_data')

    # Define transforms for image
    transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96))
        ]
    )
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])   
  
    #Define the classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    net = Classifier()
    net = net.to(device)    
 
    # Define Hyper-paramters for training loop
    max_epoch = 200
    val_interval = 1
    batch_size = 36
    lr = 1e-5

    #Define loss & optimizer & learning rate scheduler
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=lr)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=max_epoch, eta_min=1e-6)
    