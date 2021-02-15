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