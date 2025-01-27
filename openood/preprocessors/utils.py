from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .test_preprocessor import TestStandardPreProcessor
from .hamos_preprocessor import HamOSPreprocessor


def get_preprocessor(config: Config, split):
    train_preprocessors = {
        'base': BasePreprocessor,
        'hamos': HamOSPreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
    }

    if split == 'train':
        return train_preprocessors[config.preprocessor.name](config)
    else:
        try:
            return test_preprocessors[config.preprocessor.name](config)
        except KeyError:
            return test_preprocessors['base'](config)
