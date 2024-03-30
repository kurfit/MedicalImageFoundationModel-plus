
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:25:34 2022

@author: sun
"""

import os
import sys
import pathlib
import json
import time
import torch
import fnmatch
import argparse
import matplotlib.pyplot as plt


from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset, CacheDataset, PersistentDataset, SmartCacheDataset, DistributedSampler
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    Spacingd,
    CropForegroundd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    CopyItemsd,
    Resized,
    EnsureTyped,
    SplitDimd,
    ConcatItemsd,
    DeleteItemsd
)
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def main_worker(args):
    
    if args.local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f

    dist.init_process_group(backend="nccl",init_method="env://")
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    
    train_data_path = args.traindir
    test_data_path = args.testdir
    logdir_path = args.logdir
    
    train_data = list()
    test_data = list()
    for path,dirs,files in os.walk(train_data_path):
        for f in fnmatch.filter(files, '*.nii.gz'):
            train_data.append({'img':os.path.join(path,f)})
            
    for path,dirs,files in os.walk(test_data_path):
        for f in fnmatch.filter(files, '*.nii.gz'):
            test_data.append({'img':os.path.join(path,f)})
            
    print('Total Number of Training Data Samples: {}'.format(len(train_data)))      
    print('-' * 50)
    print('Total Number of Validation Data Samples: {}'.format(len(test_data)))
    print('-' * 50)
    
    holes=6000
    block_size=4
    prob=1
    
    transform = Compose([
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        Resized(keys=["img"],spatial_size=(96,96,96)),
        ScaleIntensityd(keys=["img"],minv=0, maxv=1, channel_wise=True),
        CopyItemsd(keys=["img"], times=3, names=["img_1","img_2","gt_img"], allow_missing_keys=False),
        SplitDimd(keys=["img_1","img_2"]),
        
        RandCoarseDropoutd(keys=["img_1_0"], prob=prob, holes=holes, spatial_size=block_size, dropout_holes=True, fill_value=0),
        RandCoarseDropoutd(keys=["img_1_1"], prob=prob, holes=holes, spatial_size=block_size, dropout_holes=True, fill_value=0),
        RandCoarseDropoutd(keys=["img_1_2"], prob=prob, holes=holes, spatial_size=block_size, dropout_holes=True, fill_value=0),
        RandCoarseDropoutd(keys=["img_1_3"], prob=prob, holes=holes, spatial_size=block_size, dropout_holes=True, fill_value=0),
        
        ConcatItemsd(keys=["img_1_0","img_1_1","img_1_2","img_1_3"],name='img_drop_1',dim=0),
        
        RandCoarseDropoutd(keys=["img_2_0"], prob=prob, holes=holes, spatial_size=block_size, dropout_holes=True, fill_value=0),
        RandCoarseDropoutd(keys=["img_2_1"], prob=prob, holes=holes, spatial_size=block_size, dropout_holes=True, fill_value=0),
        RandCoarseDropoutd(keys=["img_2_2"], prob=prob, holes=holes, spatial_size=block_size, dropout_holes=True, fill_value=0),
        RandCoarseDropoutd(keys=["img_2_3"], prob=prob, holes=holes, spatial_size=block_size, dropout_holes=True, fill_value=0),
        
        ConcatItemsd(keys=["img_2_0","img_2_1","img_2_2","img_2_3"],name='img_drop_2',dim=0),             
        
        DeleteItemsd(keys=["img","img_1","img_2","img_1_0","img_1_1","img_1_2","img_1_3","img_2_0","img_2_1","img_2_2","img_2_3"]),       
      
        EnsureTyped(keys=["gt_img","img_drop_1","img_drop_2"])
        
        ])
        
    batch_size = 7
    
    persistent_cache = pathlib.Path(args.cachedir)
    persistent_cache.mkdir(parents=True, exist_ok=True)
    
    
    train_ds = PersistentDataset(data=train_data, transform=transform, cache_dir=persistent_cache)
    train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2, sampler=train_sampler)
    
    val_ds = CacheDataset(data=test_data,transform=transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=2)
    