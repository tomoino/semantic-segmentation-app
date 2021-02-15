"""Trainer Class"""
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.logger import get_logger

LOG = get_logger(__name__)

class Trainer:
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model = kwargs['model']
        self.trainloader, self.testloader = kwargs['dataloaders']
        self.epochs = kwargs['epochs']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.metric = kwargs['metrics']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']
        self.ckpt_dir = kwargs['ckpt_dir']
        self.writer = SummaryWriter(str(kwargs['summary_dir']))
        self.img_size = kwargs['img_size']
        self.vis_img = kwargs['vis_img']
        self.img_outdir = kwargs['img_outdir']

    def train(self):
        best_test_iou = 0

        for epoch in range(self.epochs):
            LOG.info(f'\n==================== Epoch: {epoch} ====================')
            LOG.info('\n Train:')
            self.model.train()

            train_loss = 0

            with tqdm(self.trainloader, ncols=100) as pbar:
                for idx, (inputs, targets, img_paths) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)['out']

                    loss = self.criterion(outputs, targets.long())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    preds = torch.softmax(outputs, 1).max(1)[1]

                    ### metrics update
                    self.metric.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())
                    train_loss = self.metric.loss

                    ### logging train loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(train_loss/(idx+1))))

            # save ckpt
            if epoch != 0 and epoch % self.save_ckpt_interval == 0:
                LOG.info(' Saving Checkpoint...')
                self._save_ckpt(epoch, train_loss / (idx + 1))
                
            self.metric.result(epoch, idx, mode='train')
            self._write_summary(epoch, mode='train')
            self.metric.reset_states()

            # eval
            test_loss, test_mean_iou = self.eval(epoch)

            if test_mean_iou > best_test_iou:
                print(f'\nsaving best checkpoint (epoch: {epoch})...')
                best_test_iou = test_mean_iou
                self._save_ckpt(epoch, train_loss, mode='best')

    def eval(self, epoch, inference=False):
        self.model.eval()
        LOG.info('\n Evaluation:')

        img_path_list = []

        with torch.no_grad():
            with tqdm(self.testloader, ncols=100) as pbar:
                for idx, (inputs, targets, img_paths) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)['out']

                    loss = self.criterion(outputs, targets.long())
                    self.optimizer.zero_grad()

                    img_path_list.extend(img_paths)

                    preds = torch.softmax(outputs, 1).max(1)[1]
                    self.metric.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())
                    
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(self.metric.loss/(idx+1))))
    
        self.metric.result(epoch, idx, mode='eval')
        self._write_summary(epoch, mode='eval')

        ### save result images
        if inference:
            print('\nsaving images...')
            self._save_images(img_path_list, self.metric.preds)

        test_loss = self.metric.loss
        test_mean_iou = self.metric.mean_iou
        self.metric.reset_states()

        return (test_loss, test_mean_iou)

    def _write_summary(self, epoch: int, mode: str):
        # Change mode from 'eval' to 'val' to change the display order from left to right to train and eval.
        mode = 'val' if mode == 'eval' else mode
        self.writer.add_scalar(f'loss/{mode}', self.metric.loss, epoch)
        self.writer.add_scalar(f'iou/{mode}', self.metric.mean_iou, epoch)

    def _save_ckpt(self, epoch, loss, mode=None, zfill=4):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
    
        if mode == 'best':
            ckpt_path = self.ckpt_dir / 'best_ckpt.pth'
        else:
            ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

        torch.save({
            'epoch': epoch,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, ckpt_path)

    def _save_images(self, img_paths, preds):
        """Save Image
        Parameters
        ----------
        img_paths : list
            original image paths
        preds : tensor
            [1, 21, img_size, img_size] ([mini-batch, n_classes, height, width])
        """

        for i, img_path in enumerate(img_paths):
            # preds[i] has background label 0, so exclude background class
            pred = preds[i]

            annotated_img = self.vis_img.decode_segmap(pred)

            width = Image.open(img_path).size[0]
            height = Image.open(img_path).size[1]

            annotated_img = annotated_img.resize((width, height), Image.NEAREST)

            outpath = self.img_outdir / Path(img_path).name
            self.vis_img.save_img(annotated_img, outpath)