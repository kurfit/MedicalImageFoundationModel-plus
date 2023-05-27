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
from monai