import torch.utils.data as data
from PIL import Image
import numpy as np

# Datasetクラス
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