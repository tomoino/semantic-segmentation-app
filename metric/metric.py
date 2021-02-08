import numpy as np

class Metrics(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

        self.initialize()

    def initialize(self):
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