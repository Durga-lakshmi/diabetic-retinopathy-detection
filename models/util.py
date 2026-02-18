import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from collections import OrderedDict
import torch
import torch.nn as nn


from .basic_cnn.model import BasicCNN
from .dense.model import Dense121
from .dense.model import get_densenet121
from .convnext.model import ConvNeXt
from .efficient.model import EfficientNet

from .drnet.model import DRNet 
from .small_drnet.model import SmallDRNet
from .resnet_transfer.model import ResNet18Binary
from .resnet_multiclass.model import ResNet18Multiclass

def get_model(cfg):
    if cfg.model.name == 'BasicCNN':
        model = BasicCNN(cfg.model)

   
    elif cfg.model.name == 'Dense121':  #in ['densenet121', 'efficientnet_b3', 'resnet50']:
        #model = Dense121(cfg.model)
        model = get_densenet121(cfg.model.num_classes) 
        #for _ in model.parameters():
        #    _.requires_grad = False # freeze backbone


    elif cfg.model.name == 'convnext':
        model = ConvNeXt(cfg.model)
        #for _ in model.parameters():
        #    _.requires_grad = False # freeze backbone

   
    elif cfg.model.name == 'efficientnet':
        model = EfficientNet(cfg.model)
        #for _ in model.parameters():
        #    _.requires_grad = False # freeze backbone




    elif cfg.model.name == 'DRNet':  
        model = DRNet(num_classes=cfg.model.num_classes)
        print(f'Model "{cfg.model.name}" summary:')
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                print(f"{name:40s} | {num_params:,}")

    
    elif cfg.model.name == 'SmallDRNet':  
        model = SmallDRNet(num_classes=cfg.model.num_classes)
        print(f'Model "{cfg.model.name}" summary:')
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                print(f"{name:40s} | {num_params:,}")

    elif cfg.model.name == 'ResNet18Binary':  
        model = ResNet18Binary(pretrained=cfg.model.pretrained)
        print(f'Model "{cfg.model.name}" summary:')
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                print(f"{name:40s} | {num_params:,}")

    elif cfg.model.name == 'ResNet18Multiclass':  
        model = ResNet18Multiclass(num_classes=5, pretrained=cfg.model.pretrained)
        print(f'Model "{cfg.model.name}" summary:')
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                print(f"{name:40s} | {num_params:,}")


        print(f"\nTotal Trainable Parameters: {total_params:,}")

    else:
        raise ValueError(f'Model "{cfg.model.name}" unknown')

    return model

    

    
def threshold_sweep(y_true, y_pred_probs, thresholds=np.arange(0.1, 0.9, 0.01)):
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        y_pred = (y_pred_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    print(f"Best threshold: {best_threshold:.2f} | Best F1-score: {best_f1:.2f} -> for test_evaluation") #for testing
    return best_threshold, best_f1



