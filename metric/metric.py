import numpy as np


"""Metrics Class"""
from typing import List

# import csv
from pathlib import Path
# from statistics import mean

import numpy as np
# import pandas as pd

import torch

# from utils.logger import get_logger
# from metric.cmx import plot_cmx

# LOG = get_logger(__name__)
# pd.set_option('display.unicode.east_asian_width', True)

# class Metric:
#     def __init__(self, n_classes: int, classes: List, metric_dir: str, eps=1e-9):
#         self.n_classes = n_classes
#         self.classes = classes
#         self.metric_dir = Path(metric_dir)
#         self.eps = eps

#         self.loss_list = []
#         self.cmx = torch.zeros(self.n_classes, self.n_classes, dtype=torch.int64)

    # def update(self, preds, targets, loss):

    #     stacked = torch.stack((targets, preds), dim=1)
    #     for p in stacked:
    #         tl, pl = p.tolist()
    #         self.cmx[tl, pl] = self.cmx[tl, pl] + 1

    #     self.loss_list.append(loss)
        
#     def result(self, epoch: int, mode: str):
#         """Metric(acc, loss, precision, recall, f1score), Logging, Save and Plot CMX
        
#         Parameters
#         ----------
#         epoch : int
#             current epoch
#         mode : str
#             'train' or 'eval'
#         """
#         tp = torch.diag(self.cmx).to(torch.float32)
#         fp = (self.cmx.sum(axis=1) - torch.diag(self.cmx)).to(torch.float32)
#         fn = (self.cmx.sum(axis=0) - torch.diag(self.cmx)).to(torch.float32)

#         self.acc = (100.0 * torch.sum(tp) / torch.sum(self.cmx)).item()
#         self.loss = mean(self.loss_list)

#         self.precision = tp / (tp + fp + self.eps)
#         self.recall = tp / (tp + fn + self.eps)
#         self.f1score = tp / (tp + 0.5 * (fp + fn) + self.eps) # micro f1score

#         self._logging(epoch, mode)
#         self._save_csv(epoch, mode)

#         # plot confusion matrix
#         if mode == 'eval':
#             cmx_path = self.metric_dir / 'eval_cmx.png'
#             plot_cmx(self.cmx.clone().numpy(), self.classes, cmx_path)

#     def reset_states(self):
#         self.loss_list = []
#         self.cmx = torch.zeros(self.n_classes, self.n_classes, dtype=torch.int64)

#     def _logging(self, epoch: int, mode: str):
#         """Logging"""
#         LOG.info(f'{mode} metrics...')
#         LOG.info(f'loss:         {self.loss}')
#         LOG.info(f'accuracy:     {self.acc}')

#         df = pd.DataFrame(index=self.classes)
#         df['precision'] = self.precision.tolist()
#         df['recall'] = self.recall.tolist()
#         df['f1score'] = self.f1score.tolist()

#         LOG.info(f'\nmetrics values per classes: \n{df}\n')
#         LOG.info(f'precision:    {self.precision.mean()}')
#         LOG.info(f'recall:       {self.recall.mean()}')
#         LOG.info(f'mean_f1score: {self.f1score.mean()}\n') # micro mean f1score

#     def _save_csv(self, epoch: int, mode: str):
#         """Save results to csv"""
#         csv_path = self.metric_dir / f'{mode}_metric.csv'

#         if not csv_path.exists():
#             with open(csv_path, 'w') as logfile:
#                 logwriter = csv.writer(logfile, delimiter=',')
#                 logwriter.writerow(['epoch', f'{mode} loss', f'{mode} accuracy',
#                                     f'{mode} precision', f'{mode} recall', f'{mode} micro f1score'])

#         with open(csv_path, 'a') as logfile:
#             logwriter = csv.writer(logfile, delimiter=',')
#             logwriter.writerow([epoch, self.loss, self.acc, 
#                                 self.precision.mean().item(), self.recall.mean().item(), self.f1score.mean().item()])

class Metric(object):
    def __init__(self, n_classes: int, metric_dir: str):
        self.n_classes = n_classes
        self.metric_dir = Path(metric_dir)

        # self.eps = eps
        self.loss = 0
        self.pred_list = []
        self.target_list = []

        # self.cmx = torch.zeros(self.n_classes, self.n_classes, dtype=torch.int64)
        self.cmx = np.zeros((self.n_classes, self.n_classes))

    def update(self, preds, targets, loss):
        self.pred_list.append(preds)
        self.target_list.append(targets)
        self.loss += loss

    def result(self, epoch: int, idx:int, mode: str):
        """Metric(acc, loss, precision, recall, f1score), Logging, Save and Plot CMX
        
        Parameters
        ----------
        epoch : int
            current epoch
        mode : str
            'train' or 'eval'
        """

        self.preds = torch.cat([p for p in self.pred_list], axis=0)
        self.targets = torch.cat([t for t in self.target_list], axis=0)
        self.calc_metrics(self.preds, self.targets, self.loss/(idx+1), epoch, mode=mode)
    
    def reset_states(self):
        self.loss = 0
        self.pred_list = []
        self.target_list = []
        self.cmx = np.zeros((self.n_classes, self.n_classes))

    def calc_metrics(self, preds, targets, loss, epoch, mode):
        preds = preds.view(-1)
        targets = targets.view(-1)

        preds = preds.numpy()
        targets = targets.numpy()

        # calc histgram and make confusion matrix
        cmx = np.bincount(self.n_classes * targets.astype(int) 
                         + preds, minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        
        with np.errstate(invalid='ignore'):
            self.ious = np.diag(cmx) / (cmx.sum(axis=1) + cmx.sum(axis=0) - np.diag(cmx))
        
        self.loss = loss
        self.mean_iou = np.nanmean(self.ious)
