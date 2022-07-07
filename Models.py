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
                    out_channels=4,
                    img_size=(96, 96, 96),
                    patch_size=(16, 16, 16),
                    pos_embed='conv',
                    num_heads=32,
                    n