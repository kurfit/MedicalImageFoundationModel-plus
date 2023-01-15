import os
import sys
import torch
import monai
import random
import fnmatch
import logging
import torch.nn as nn
from monai.networks.nets import ViTAutoEnc
from monai.data import DataLoader, CacheDataset, CSVSaver
from monai.transforms import (
    LoadImaged,
    Compose,
    Resized,
    ScaleIntensityd
)

from Models import Classifier

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Select the GPU

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
                data_1.append(os.path.joi