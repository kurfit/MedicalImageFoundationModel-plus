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
                    num_layers=16,
                    hidden_size=2048,
                    mlp_dim=3072
        )
        '''Load the pre-trained model parameters'''
  