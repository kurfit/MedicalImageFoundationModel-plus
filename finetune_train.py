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
        data_0_dir = os.path.join(path,'') #Name of