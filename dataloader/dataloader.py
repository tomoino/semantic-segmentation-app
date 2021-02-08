# from torch.utils.data import DataLoader

"""DataLoader class"""
from typing import List
import torch.utils.data as data
from dataloader.utils import make_data_list
from dataloader.transform import DataTransform
from dataloader.dataset import Dataset

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(dataroot: str, annotationroot: str, annotationpath: str):
        """load dataset from path
        Parameters
        ----------
        dataroot : str
            path to the image data directory e.g. './data/images/'
        annotationroot : str
            a path to the annotation directory
        annotationpath : str
            path to the annotation text e.g. './data/annotation/train.txt'
        Returns
        -------
        Tuple of list
            img_list: e.g. ['./data/images/car1.png', './data/images/dog4.png', ...]
            annot_list:  e.g. ['./data/images/car1.png', './data/images/dog4.png', ...]
        """
        img_list, annot_list = make_data_list(dataroot, annotationroot, annotationpath)
        return (img_list, annot_list)

    @staticmethod
    def preprocess_data(data_config: object, img_list: List, annot_list: List, batch_size: int, mode: str):
        """Preprocess dataset
        Parameters
        ----------
        data_config : object
            data configuration
        img_list : List
            a list of image paths
        annot_list : List
            a list of annotation
        batch_size : int
            batch_size
        mode : str
            'train' or 'eval'
        Returns
        -------
        Object : 
            DataLoader instance
        Raises
        ------
        ValueError
            raise value error if the mode is not 'train' or 'eval'
        """
        # transform
        resize = (data_config.img_size[0], data_config.img_size[1])
        color_mean = tuple(data_config.color_mean)
        color_std = tuple(data_config.color_std)
        transform = DataTransform(resize, color_mean, color_std, mode)

        # dataset
        dataset = Dataset(img_list, annot_list, transform, data_config.label_color_map)

        # dataloader
        if mode == 'train':
            return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        elif mode == 'eval':
            return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            raise ValueError('the mode should be train or eval. this mode is not supported')
