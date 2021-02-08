""" 
学習用スクリプト

args:
    --configfile:String 設定ファイルの指定
    --eval:Boolean eval mode の指定。T: 評価、 F: 学習 

"""
import argparse
import sys

from utils.load import load_yaml
from model import get_model

def parser():
    parser = argparse.ArgumentParser('Semantic Segmentation Argument')
    parser.add_argument('--configfile', type=str, default='./configs/default.yml', help='config file')
    parser.add_argument('--eval', action='store_true', help='eval mode')
    args = parser.parse_args()
    return args

def run(args):
    """Builds model, loads data, trains and evaluates"""
    config = load_yaml(args.configfile)
    model = get_model(config)
    model.load_data(args.eval)
    model.build()
    
    if args.eval:
        model.evaluate()
    else:
        model.train()

    sys.exit()

if __name__ == '__main__':
    args = parser()
    run(args)

    # ネットワークモデルの作成
    # ネットワークの定義
    # import torchvision.models as models
    # network = models.segmentation.fcn_resnet50(pretrained=False, num_classes=n_classes)

    # check GPU
    # import torch
    # if torch.cuda.is_available():
    #     print("　use gpu..")
    #     device = torch.device("cuda:0")
    # else:
    #     print("use cpu..")
    #     device = torch.device("cpu")

    # ネットワークをGPU or CPUに転送
    # network = network.to(device)
    # networkの確認
    # print(network)
    # networkの出力はOrderDict型なので、'out'をつけてoutputのみを取り出します。
    # for (inputs, targets, img_path) in train_loader:
    #     inputs = inputs.to(device)
    #     outputs = network(inputs)['out']
    #     print(outputs.shape)
    #     break

    # loss関数
    # import torch.nn as nn
    # criterion = nn.CrossEntropyLoss()

    # optimizer
    # import torch.optim as optim
    # optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-4)

    # Metrics の定義
    # from metric.metric import Metrics
    # metrics=Metrics(n_classes=n_classes)
    
    # 諸々のconfiguration

    # visualize用のインスタンス作成
    # from utils.vis_image import VisImage
    # vis_img = VisImage(n_classes=n_classes, label_color_map=label_color_map)

    # 定期的にcheckpoint(重みファイル)を保存。ここでは100と設定するので実質保存しない。
    # save_ckpt_interval = 100

    # checkpointの出力先
    # ckpt_dir = Path(ROOT_PATH) / "result" / "checkpoint"
    # Path(ckpt_dir).mkdir(exist_ok=True, parents=True)

    # イメージの出力先. この後行うテスト推論時に出力されます。
    # img_outdir = Path(ROOT_PATH) / "result" / "imgs"
    # Path(img_outdir).mkdir(exist_ok=True, parents=True)

    # kwargs = {
    #     'device': device,
    #     'network': network,
    #     'optimizer': optimizer,
    #     'criterion': criterion,
    #     'data_loaders': (train_loader, test_loader),
    #     'metrics': metrics,
    #     'vis_img': vis_img,
    #     'img_size': img_size,
    #     'save_ckpt_interval': save_ckpt_interval,
    #     'ckpt_dir': ckpt_dir,
    #     'img_outdir': img_outdir,
    # }

    # 辞書型で渡す
    from model.segmentation import SemanticSegmentation
    semantic_segmentation = SemanticSegmentation(**kwargs)

    # 学習の実行
    # logを取るためのリスト
    train_losses = []
    test_losses = []
    train_iou = []
    test_iou = []

    # 開始エポックと学習を回すエポック数。今回は1エポックでOK。
    start_epoch = 0
    epochs = 1

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss, train_mean_iou, test_loss, test_mean_iou = semantic_segmentation.train(epoch)

        # print(f"\ntrain_loss: {train_loss}, train_iou: {train_mean_iou}")
        print(f"test_loss: {test_loss}, test_iou: {test_mean_iou}")

        train_losses.append(train_loss)
        train_iou.append(train_mean_iou)
        test_losses.append(test_loss)
        test_iou.append(test_mean_iou)