from openood.utils import Config

from .base_recorder import BaseRecorder
from .hamos_recorder import HamOSRecorder


def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
        'hamos': HamOSRecorder,
    }

    return recorders[config.recorder.name](config)
