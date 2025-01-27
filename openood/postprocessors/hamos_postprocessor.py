from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


class HamOSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(HamOSPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.label_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            label_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    labels = batch['label'].cpu().numpy()  # Assuming labels are in batch['label']

                    logit, feature = net.forward(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())
                    label_log.append(labels)

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.label_log = np.concatenate(label_log, axis=0)
            self.index = faiss.IndexFlatL2(feature.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, feature = net.forward(data, return_feature=True)
        D, I = self.index.search(
            feature.cpu().numpy(),  # feature is already normalized within net
            self.K,
        )
        kth_dist = -D[:, -1]
        nearest_labels = self.label_log[I]
        prob = torch.softmax(logits, dim=1)
        conf, pred = torch.max(prob, dim=1)

        return pred, torch.from_numpy(kth_dist).cuda()
    
    @torch.no_grad()
    def postprocess_feat(self, data: Any):
        D, I = self.index.search(
            data,  # feature is already normalized within net
            self.K,
        )
        kth_dist = -D[:, -1]
        return kth_dist

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
