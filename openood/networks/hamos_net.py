import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class HamOSNet(nn.Module):
    def __init__(self, backbone, head, feat_dim, num_classes):
        super(HamOSNet, self).__init__()

        self.backbone = backbone

        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.prototypes = nn.Parameter(torch.zeros(num_classes, feat_dim),
                                       requires_grad=True)

        if head == 'linear':
            self.head = nn.Linear(feature_size, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(nn.Linear(feature_size, feature_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(feature_size, feat_dim))

    def forward(self, x, return_feature=False):
        if return_feature:
            logits, feat = self.backbone.forward(x, return_feature=True)
            feat = feat.squeeze()

            unnorm_features = self.head(feat)
            features = F.normalize(unnorm_features, dim=1)

            return logits, features
        else:
            return self.backbone(x)

    def intermediate_forward(self, x):
        logits, feat = self.backbone.forward(x, return_feature=True)
        feat = feat.squeeze()
        return F.normalize(feat, dim=1)
    
    def get_fc(self):
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.backbone.fc
