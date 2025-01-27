from openood.utils import Config

from .finetune_pipeline import FinetunePipeline
from .test_acc_pipeline import TestAccPipeline
from .test_ood_pipeline import TestOODPipeline
from .train_oe_pipeline import TrainOEPipeline
from .train_pipeline import TrainPipeline
from .test_ood_pipeline_aps import TestOODPipelineAPS


def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'finetune': FinetunePipeline,
        'test_acc': TestAccPipeline,
        'test_ood': TestOODPipeline,
        'train_oe': TrainOEPipeline,
        'test_ood_aps': TestOODPipelineAPS
    }

    return pipelines[config.pipeline.name](config)
