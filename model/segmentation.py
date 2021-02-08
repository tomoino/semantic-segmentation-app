from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

class SemanticSegmentation(object):
    def __init__(self, **kwargs):
        # 辞書型で受け取ります
        self.device = kwargs['device']
        self.network = kwargs['network']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.train_loader, self.test_loader = kwargs['data_loaders']
        self.metrics = kwargs['metrics']
        self.vis_img = kwargs['vis_img']
        self.img_size = kwargs['img_size']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']
        self.ckpt_dir = kwargs['ckpt_dir']
        self.img_outdir = kwargs['img_outdir']

    # 学習用関数の定義
    def train(self, epoch):

        best_test_iou = 0
        print(f'\n\n==================== Epoch: {epoch} ====================')
        print('### train:')
        self.network.train()

        train_loss = 0
        pred_list = []
        target_list = []

        with tqdm(self.train_loader, ncols=100) as pbar:
            for idx, (inputs, targets, img_paths_) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.network(inputs)['out']

                loss = self.criterion(outputs, targets.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                preds = torch.softmax(outputs, 1).max(1)[1]

                ### metrics update
                pred_list.append(preds.cpu().detach().clone())
                target_list.append(targets.cpu().detach().clone())

                ### logging train loss and accuracy
                pbar.set_postfix(OrderedDict(
                    epoch="{:>10}".format(epoch),
                    loss="{:.4f}".format(train_loss/(idx+1))))

        if epoch % self.save_ckpt_interval == 0:
            print('\nsaving checkpoint...')
            self._save_ckpt(epoch, train_loss/(idx+1))

        print('\ncalculate metrics...')
        preds = torch.cat([p for p in pred_list], axis=0)
        targets = torch.cat([t for t in target_list], axis=0)
        self.metrics.calc_metrics(preds, targets, train_loss/(idx+1), epoch, mode='train')
        train_mean_iou = self.metrics.mean_iou
        train_loss = self.metrics.loss
        self.metrics.initialize()

        ### test
        print('\n### test:')
        test_mean_iou, test_loss = self.test(epoch)

        if test_mean_iou > best_test_iou:
            print(f'\nsaving best checkpoint (epoch: {epoch})...')
            best_test_iou = test_mean_iou
            self._save_ckpt(epoch, train_loss, mode='best')

        return (train_loss, train_mean_iou, test_loss, test_mean_iou)

    def test(self, epoch, inference=False):
        self.network.eval()

        test_loss = 0
        img_path_list = []
        pred_list = []
        target_list = []

        with torch.no_grad():
            with tqdm(self.test_loader, ncols=100) as pbar:
                for idx, (inputs, targets, img_paths) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.network(inputs)['out']

                    loss = self.criterion(outputs, targets.long())

                    test_loss += loss.item()

                    img_path_list.extend(img_paths)

                    preds = torch.softmax(outputs, 1).max(1)[1]

                    ### metrics update
                    pred_list.append(preds.cpu().detach().clone())
                    target_list.append(targets.cpu().detach().clone())

                    ### logging test loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(test_loss/(idx+1))))

            ### metrics
            print('\ncalculate metrics...')
            preds = torch.cat([p for p in pred_list], axis=0)
            targets = torch.cat([t for t in target_list], axis=0)
            self.metrics.calc_metrics(preds, targets, test_loss/(idx+1), epoch, mode='test')
            test_mean_iou = self.metrics.mean_iou
            test_loss = self.metrics.loss

            ### save result images
            if inference:
                print('\nsaving images...')
                self._save_images(img_path_list, preds)

            self.metrics.initialize()

        return test_mean_iou, test_loss

    def _save_ckpt(self, epoch, loss, mode=None, zfill=4):
        if isinstance(self.network, nn.DataParallel):
            network = self.network.module
        else:
            network = self.network

        if mode == 'best':
            ckpt_path = self.ckpt_dir / "ckpt_semanticsegmentation_basic_fcn_res50.pth"
        else:
            ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

        torch.save({
            'epoch': epoch,
            'network': network,
            'model_state_dict': network.state_dict(),
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