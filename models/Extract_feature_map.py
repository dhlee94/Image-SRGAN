import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:8])

    def forward(self, x):
        if x.size()[1] == 1:
          x = x.repeat(1, 3, 1, 1)
        return self.feature_extractor(x)