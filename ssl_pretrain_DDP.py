
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