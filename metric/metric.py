"""Metrics Class"""
from typing import List
from pathlib import Path
import numpy as np
import torch

class Metric(object):
    def __init__(self, n_classes: int, metric_dir: str):
        self.n_classes = n_classes
        self.metric_dir = Path(metric_dir)

        self.loss = 0
        self.pred_list = []
        self.target_list = []

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