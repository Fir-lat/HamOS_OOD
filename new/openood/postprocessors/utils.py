from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .hamos_postprocessor import HamOSPostprocessor

def get_postprocessor(config: Config):
    postprocessors = {
        'msp': BasePostprocessor,
        'hamos': HamOSPostprocessor
    }

    return postprocessors[config.postprocessor.name](config)
