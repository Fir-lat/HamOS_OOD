from typing import Callable, List, Type
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor

import openood.utils.comm as comm



class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = './data',
        config_root: str = './configs',
        preprocessor: Callable = None,
        postprocessor_name: str = None,
        postprocessor: Type[BasePostprocessor] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        # check the arguments
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print(
                'Postprocessor_name is ignored because postprocessor is passed'
            )
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(config_root, postprocessor_name,
                                              id_name)
        if not isinstance(postprocessor, BasePostprocessor):
            raise TypeError(
                'postprocessor should inherit BasePostprocessor in OpenOOD')

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, **loader_kwargs)

        # wrap base model to work with certain postprocessors

        # postprocessor setup
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])

        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                {k: None
                 for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        # perform hyperparameter search if have not done so
        if (self.postprocessor.APS_mode
                and not self.postprocessor.hyperparam_search_done):
            self.hyperparam_search()

        self.net.eval()

        # how to ensure the postprocessors can work with
        # models whose definition doesn't align with OpenOOD

    def _classifier_inference(self,
                              data_loader: DataLoader,
                              msg: str = 'Acc Eval',
                              progress: bool = True):
        self.net.eval()

        
        id_pred, id_conf, id_gt = self.postprocessor.inference(self.net, data_loader, progress)

        return id_pred, id_gt

    def eval_acc(self, data_name: str = 'id') -> float:
        if data_name == 'id':
            if self.metrics['id_acc'] is not None:
                return self.metrics['id_acc']
            else:
                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics['id_acc'] = acc
                return acc
        elif data_name == 'csid':
            if self.metrics['csid_acc'] is not None:
                return self.metrics['csid_acc']
            else:
                correct, total = 0, 0
                for _, (dataname, dataloader) in enumerate(
                        self.dataloader_dict['csid'].items()):
                    if self.scores['csid_preds'][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader, f'CSID {dataname} Acc Eval')
                        self.scores['csid_preds'][dataname] = all_preds
                        self.scores['csid_labels'][dataname] = all_labels
                    else:
                        all_preds = self.scores['csid_preds'][dataname]
                        all_labels = self.scores['csid_labels'][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                correct += (all_preds == all_labels).sum().item()
                total += len(all_labels)

                acc = correct / total * 100
                self.metrics['csid_acc'] = acc
                return acc
        else:
            raise ValueError(f'Unknown data name {data_name}')

    def eval_ood(self, fsood: bool = False, progress: bool = True):
        id_name = 'id' if not fsood else 'csid'
        task = 'ood' if not fsood else 'fsood'
        if self.metrics[task] is None:
            self.net.eval()

            # id score
            if self.scores['id']['test'] is None:
                print(f'Performing inference on {self.id_name} test set...',
                      flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress)
                self.scores['id']['test'] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']

            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][dataset_name] is None:
                        print(
                            f'Performing inference on {self.id_name} '
                            f'(cs) test set [{i+1}]: {dataset_name}...',
                            flush=True)
                        temp_pred, temp_conf, temp_gt = \
                            self.postprocessor.inference(
                                self.net,
                                self.dataloader_dict['csid'][dataset_name],
                                progress)
                        self.scores['csid'][dataset_name] = [
                            temp_pred, temp_conf, temp_gt
                        ]

                    csid_pred.append(self.scores['csid'][dataset_name][0])
                    csid_conf.append(self.scores['csid'][dataset_name][1])
                    csid_gt.append(self.scores['csid'][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # load nearood data and compute ood metrics
            near_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                          ood_split='near',
                                          progress=progress)
            # load farood data and compute ood metrics
            far_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                         ood_split='far',
                                         progress=progress)

            if self.metrics[f'{id_name}_acc'] is None:
                self.eval_acc(id_name)
            near_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
                                           len(near_metrics))
            far_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
                                          len(far_metrics))

            self.metrics[task] = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict['ood']['near'].keys()) +
                ['nearood'] + list(self.dataloader_dict['ood']['far'].keys()) +
                ['farood'],
                columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC'],
            )
        else:
            print('Evaluation has already been done!')

        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.float_format',
                '{:,.2f}'.format):  # more options can be specified also
            print(self.metrics[task])

        return self.metrics[task]
    
    def eval_ood_degree(self, fsood: bool = False, progress: bool = True):

        print(f'Performing inference on {self.id_name} test set...', flush=True)
        activation_log_tensor = torch.tensor(self.postprocessor.activation_log, dtype=torch.float32).cuda()
        label_log_tensor = torch.tensor(self.postprocessor.label_log, dtype=torch.long).cuda()
        n_cls = torch.max(label_log_tensor) + 1
        centroids = torch.zeros((n_cls, activation_log_tensor.shape[1]), dtype=torch.float32).cuda()
        for i in range(n_cls):
            class_indices = (label_log_tensor == i)
            class_embeddings = activation_log_tensor[class_indices]
            centroids[i] = torch.mean(class_embeddings, dim=0)
            centroids[i] /= torch.norm(centroids[i])
        
        similarities = torch.matmul(centroids, centroids.T)
        # inter-class mean cosine similarity
        inter_mean_cos_sim = torch.acos(torch.mean(similarities[torch.triu(torch.ones(n_cls, n_cls), diagonal=1) == 1])).cpu().numpy() * (180 / math.pi)

        # intra-class mean cosine similarity
        id_test_cos_sim = []
        with torch.no_grad():
            for batch in tqdm(self.dataloader_dict['id']['test'], disable=not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                _, feat = self.net.forward(data, return_feature=True)
                cos_sim = torch.matmul(feat, centroids[label].T)
                id_test_cos_sim.append(cos_sim)
        id_test_cos_sim = torch.cat(id_test_cos_sim, dim=0)
        intra_mean_cos_sim = torch.acos(torch.mean(id_test_cos_sim)).cpu().numpy() * (180 / math.pi)

        far_metrics = {}
        for dataset_name, ood_dl in self.dataloader_dict['ood']['far'].items():
            # if dataset_name not in ['lsun']: continue
            far_metrics[dataset_name] = []
            with torch.no_grad():
                for batch in tqdm(ood_dl, disable=not comm.is_main_process()):
                    data = batch['data'].cuda()
                    _, feat = self.net.forward(data, return_feature=True)
                    cos_sim = torch.matmul(feat, centroids.T)
                    cos_sim_max, _ = torch.max(cos_sim, dim=1)
                    far_metrics[dataset_name].append(cos_sim_max)
            far_metrics[dataset_name] = torch.acos(torch.mean(torch.cat(far_metrics[dataset_name], dim=0))).cpu().numpy() * (180 / math.pi)

        far_metrics['far'] = np.mean(list(far_metrics.values()))
        print(inter_mean_cos_sim, intra_mean_cos_sim, far_metrics)
        # return None
        degree_metrics = pd.DataFrame(
            np.concatenate([[inter_mean_cos_sim], [intra_mean_cos_sim], list(far_metrics.values())], axis=0),
            index=['inter'] + ['intra'] + list(self.dataloader_dict['ood']['far'].keys()) + ['farood'],
            columns=['Degree'],
        )
        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.float_format',
                '{:,.2f}'.format):  # more options can be specified also
            print(degree_metrics)
        
        return degree_metrics

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True):
        print(f'Processing {ood_split} ood...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict['ood'][
                ood_split].items():
            if self.scores['ood'][ood_split][dataset_name] is None:
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress)
                self.scores['ood'][ood_split][dataset_name] = [
                    ood_pred, ood_conf, ood_gt
                ]
            else:
                print(
                    'Inference has been performed on '
                    f'{dataset_name} dataset...',
                    flush=True)
                [ood_pred, ood_conf,
                 ood_gt] = self.scores['ood'][ood_split][dataset_name]

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)
            self._print_metrics(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        self._print_metrics(list(metrics_mean[0]))
        return np.concatenate([metrics_list, metrics_mean], axis=0) * 100

    def _print_metrics(self, metrics):
        [fpr, auroc, aupr_in, aupr_out, _] = metrics

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'])
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'])

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            print('Hyperparam: {}, auroc: {}'.format(hyperparam, auroc))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('Final hyperparam: {}'.format(
            self.postprocessor.get_hyperparam()))
        self.postprocessor.hyperparam_search_done = True

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
