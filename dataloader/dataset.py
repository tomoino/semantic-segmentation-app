from pathlib import Path

def make_datapath_list(rootpath, train_data='train.txt', test_data='test.txt', img_extension='.jpg', anno_extension='.png'):
    """
    Create list of image and annotation data path
    Parameters
    ----------
    rootpath : str
        path to the data directory
    train_data : str
        text file with train filename
    test_data : str
        text file with test filename
    img_extension : str
        extension of image
    anno_extension : str
        extension of annotation
    Returns
    ----------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
    """

    img_dir = Path(rootpath) / 'JPEGImages'
    annot_dir = Path(rootpath) / 'SegmentationClass'

    train_filenames = Path(rootpath) / 'ImageSets' / 'Segmentation' / train_data
    test_filenames = Path(rootpath) / 'ImageSets' / 'Segmentation' / test_data

    # create train img and annot path list
    train_img_list = []
    train_annot_list = []

    for line in open(train_filenames):
        line = line.rstrip('\n')
        img_fname = line + img_extension
        img_path = img_dir / img_fname
        anno_fname = line + anno_extension
        annot_path = annot_dir / anno_fname
        train_img_list.append(str(img_path))
        train_annot_list.append(str(annot_path))

    # create test img and annot path list
    test_img_list = []
    test_annot_list = []

    for line in open(test_filenames):
        line = line.rstrip('\n')
        img_fname = line + img_extension
        img_path = img_dir / img_fname
        anno_fname = line + anno_extension
        annot_path = annot_dir / anno_fname
        test_img_list.append(str(img_path))
        test_annot_list.append(str(annot_path))

    return train_img_list, train_annot_list, test_img_list, test_annot_list

import torch.utils.data as data
from PIL import Image
import numpy as np
# VOC用のDatasetクラス
class Dataset(data.Dataset):
    def __init__(self, img_list, anno_list, transform, label_color_map):
        self.img_list = img_list
        self.anno_list = anno_list
        self.transform = transform
        self.label_color_map = label_color_map # list [[]]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno, img_filepath = self.pull_item(index)
        return (img, anno, img_filepath)

    def pull_item(self, index):
        
        img_filepath = self.img_list[index]
        img = Image.open(img_filepath)

        anno_filepath = self.anno_list[index]
        anno = Image.open(anno_filepath).convert("RGB")
        anno = Image.fromarray(self.encode_segmap(np.array(anno)))
        img, anno = self.transform(img, anno)

        return img, anno, img_filepath

    # label(アノテーション)データは、RGBの画像になっている
    # それを0~20までの値でできたGrayScaleの画像に変換するための処理
    def encode_segmap(self, mask):

        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for ii, label in enumerate(np.asarray(self.label_color_map)):
            label_mask[np.where(np.all(mask==label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(np.uint8)
        return label_mask