# import mmcv
from copy import deepcopy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from mmcls.apis import init_model

import openood.utils.comm as comm

from .densenet import DenseNet3
from .resnet18_32x32 import ResNet18_32x32, ResNet34_32x32
from .resnet18_64x64 import ResNet18_64x64
from .resnet18_224x224 import ResNet18_224x224, ResNet50_224x224
from .resnet18_256x256 import ResNet18_256x256
from .resnet50 import ResNet50
from .vit_b_16 import ViT_B_16
from .wrn import WideResNet
from .hamos_net import HamOSNet


def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)

    elif network_config.name == 'resnet34_32x32':
        net = ResNet34_32x32(num_classes=num_classes)

    elif network_config.name == 'resnet18_256x256':
        net = ResNet18_256x256(num_classes=num_classes)

    elif network_config.name == 'resnet18_64x64':
        net = ResNet18_64x64(num_classes=num_classes)

    elif network_config.name == 'resnet18_224x224':
        net = ResNet18_224x224(num_classes=num_classes)

    elif network_config.name == 'resnet50_224x224':
        net = ResNet50_224x224(num_classes=num_classes)

    elif network_config.name == 'resnet50':
        net = ResNet50(num_classes=num_classes)

    elif network_config.name == 'wrn':
        net = WideResNet(depth=28,
                         widen_factor=10,
                         dropRate=0.0,
                         num_classes=num_classes)

    elif network_config.name == 'densenet':
        net = DenseNet3(depth=100,
                        growth_rate=12,
                        reduction=0.5,
                        bottleneck=True,
                        dropRate=0.0,
                        num_classes=num_classes)

    elif network_config.name == 'hamos_net':
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)

        net = HamOSNet(backbone=backbone,
                      head=network_config.head,
                      feat_dim=network_config.feat_dim,
                      num_classes=num_classes)

    elif network_config.name == 'bit':
        net = KNOWN_MODELS[network_config.model](
            head_size=network_config.num_logits,
            zero_head=True,
            num_block_open=network_config.num_block_open)

    elif network_config.name == 'vit-b-16':
        net = ViT_B_16(num_classes=num_classes)

    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        if type(net) is dict:
            if isinstance(network_config.checkpoint, list):
                for subnet, checkpoint in zip(net.values(),
                                              network_config.checkpoint):
                    if checkpoint is not None:
                        if checkpoint != 'none':
                            subnet.load_state_dict(torch.load(checkpoint),
                                                   strict=False)
            elif isinstance(network_config.checkpoint, str):
                ckpt = torch.load(network_config.checkpoint)
                subnet_ckpts = {k: {} for k in net.keys()}
                for k, v in ckpt.items():
                    for subnet_name in net.keys():
                        if k.startswith(subnet_name):
                            subnet_ckpts[subnet_name][k.replace(
                                subnet_name + '.', '')] = v
                            break
                if 'dummy_net' in net:
                    subnet_ckpts['dummy_net'] = ckpt

                for subnet_name, subnet in net.items():
                    subnet.load_state_dict(subnet_ckpts[subnet_name])

        elif network_config.name == 'bit' and not network_config.normal_load:
            net.load_from(np.load(network_config.checkpoint))
        elif network_config.name == 'vit':
            pass
        else:
            try:
                net.load_state_dict(torch.load(network_config.checkpoint),
                                    strict=False)
            except RuntimeError:
                # sometimes fc should not be loaded
                loaded_pth = torch.load(network_config.checkpoint)
                loaded_pth.pop('fc.weight')
                loaded_pth.pop('fc.bias')
                net.load_state_dict(loaded_pth, strict=False)
        print('Model Loading {} Completed!'.format(network_config.name))

    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet.cuda(),
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()

    cudnn.benchmark = True
    return net
