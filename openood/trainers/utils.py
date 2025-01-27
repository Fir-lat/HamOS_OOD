from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .hamos_trainer import HamOSTrainer


def get_trainer(net, train_loader: DataLoader, val_loader: DataLoader,
                config: Config):
    if type(train_loader) is DataLoader:
        trainers = {
            'base': BaseTrainer,
            'hamos': HamOSTrainer
        }
        if config.trainer.name in ['hamos']:
            return trainers[config.trainer.name](net, train_loader, val_loader, config)
        else:
            return trainers[config.trainer.name](net, train_loader, config)

    else:
        raise ValueError("Unsupported Trainer!")
