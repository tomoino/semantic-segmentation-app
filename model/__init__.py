# """Models"""

# from typing import Dict

# from configs.supported_model import SUPPORTED_MODEL

# from .fcn_resnet import FCNResNet

# def get_model(config: Dict) -> object:
#     """Get Model Class"""
#     model_name = config['model']['name']

#     if model_name in SUPPORTED_MODEL['FCNResNet']:
#         return FCNResNet(config)
#     else:
#         NotImplementedError('The model is not supported.')