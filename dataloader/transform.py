from PIL import Image
from torchvision import transforms
import numpy as np
import torch

# Compose
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img

# Resize
class Resize(object):
    def __init__(self, resize: tuple):
        self.resize = resize

    def __call__(self, img, anno_class_img):

        # width = img.size[0]  # img.size=[幅][高さ]
        # height = img.size[1]  # img.size=[幅][高さ]

        img = img.resize(self.resize,
                         Image.BICUBIC)
        anno_class_img = anno_class_img.resize(
            self.resize, Image.NEAREST)

        return img, anno_class_img

# Normalize
class Normalize_Tensor(object):
    def __init__(self, color_mean: tuple, color_std: tuple):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):

        # PIL画像をTensorに。大きさは最大1に規格化される
        img = transforms.functional.to_tensor(img)

        # 色情報の標準化
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        # アノテーション画像をNumpyに変換
        anno_class_img = np.array(anno_class_img)  # [高さ][幅]

        # 'ambigious'には255が格納されているので、0の背景にしておく
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # アノテーション画像をTensorに
        anno_class_img = torch.from_numpy(anno_class_img)

        return img, anno_class_img

class DataTransform():
    def __init__(self, resize, color_mean, color_std, mode):
        if mode == 'train':
            self.data_transform = Compose([
                Resize(resize),
                Normalize_Tensor(color_mean, color_std)
            ])
        elif mode == 'test':
            self.data_transform = Compose([
                Resize(resize),
                Normalize_Tensor(color_mean, color_std)
            ])

    def __call__(self, img, anno_class_img):
        return self.data_transform(img, anno_class_img)