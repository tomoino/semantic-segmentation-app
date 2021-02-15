"""Inferrer"""
from typing import Dict
from PIL import Image

import torch
import torch.nn.functional as F

from utils.config import Config
from utils.load import load_yaml
from model import get_model
from model.common.device import setup_device
from dataloader.transform import DataTransform

from executor.vis_image import VisImage

class Inferrer:
    def __init__(self, configfile: str):
        # Config
        config = load_yaml(configfile)
        self.config = Config.from_json(config)

        # Builds model
        self.model = get_model(config)
        self.model.build()
        self.model_name = self.model.model_name

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.model.to(self.device)
        self.model.model.eval()

        self.n_classes = self.config.model.n_classes
        self.vis_img = VisImage(n_classes=self.n_classes, label_color_map=self.config.data.label_color_map)

    def preprocess(self, image: Image) -> torch.Tensor:
        """Preprocess Image
        PIL.Image to Tensor
        """
        resize = (self.config.data.img_size[0], self.config.data.img_size[1])
        color_mean = tuple(self.config.data.color_mean)
        color_std = tuple(self.config.data.color_std)
        transform = DataTransform(resize, color_mean, color_std, mode='eval')
        image, _ = transform(image, image)
        image = image.unsqueeze(0)
        return image

    def infer(self, image: Image = None) -> Dict:
        """Infer an image
        Parameters
        ----------
        image : PIL.Image, optional
            input image, by default None
        Returns
        -------
        image : PIL.Image
            annotated image
        """

        shape = image.size
        tensor_image = self.preprocess(image)
        with torch.no_grad():
            tensor_image = tensor_image.to(self.device)
            output = self.model.model(tensor_image)['out']
            pred = torch.softmax(output, 1).max(1)[1]
            pred = torch.cat([pred.cpu().detach().clone()], axis=0)
            annotated_img = self._make_annotated_image(shape, pred)
            
            # mask = Image.new("RGBA", shape[0], shape[1])
            # image = Image.composite(image, annotated_img, mask)
            result = Image.blend(image, annotated_img, 0.7)
            return result

    def _make_annotated_image(self, shape, pred):
            annotated_img = self.vis_img.decode_segmap(pred[0])

            width = shape[0]
            height = shape[1]

            annotated_img = annotated_img.resize((width, height), Image.NEAREST)

            return annotated_img
