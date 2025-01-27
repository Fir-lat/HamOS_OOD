import os
import urllib.request

from openood.postprocessors import BasePostprocessor, HamOSPostprocessor
from openood.utils.config import Config, merge_configs

postprocessors = {
    'msp': BasePostprocessor,
    'hamos': HamOSPostprocessor,
}


def get_postprocessor(config_root: str, postprocessor_name: str,
                      id_data_name: str):
    postprocessor_config_path = os.path.join(config_root, 'postprocessors',
                                             f'{postprocessor_name}.yml')
    if not os.path.exists(postprocessor_config_path):
        raise ValueError("Post processors doesn't exist")

    config = Config(postprocessor_config_path)
    config = merge_configs(config,
                           Config(**{'dataset': {
                               'name': id_data_name
                           }}))
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor
