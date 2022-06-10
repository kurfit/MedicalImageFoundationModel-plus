import os
import torch
import torch.nn as nn
from collections import OrderedDict
from monai.networks.nets import ViTAutoEnc

class Classifier(nn.Module):    
    def __init__(self):
        super().__init__()
        self.model = ViTAutoEnc(
                    in_channels=4,
           