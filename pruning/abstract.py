from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd

import os, json, pickle

import torch

from .mask import mask_module
from .modules import MaskedModule,masked_modules
from .utils import get_params,check_fraction
from torch import nn


class Pruning(ABC):

    def __init__(self, model,config=None,dataset_name=None, compression=None,device=None,log=None):
        self.keep_ratio = check_fraction(compression)
        self.model = model
        self.importances = dict()
        self.dataset_name=dataset_name
        self.config=config
        self.device=device
        self.masks=dict()
        self.log=log

    @abstractmethod
    def model_masks(self):
        """Compute masks for a given model

        """
        # TODO
        pass
        # return masks
    def init_mask(self):
        self._init_mask(self.model)
    def _init_mask(self,model):
        new_children = {}
        for name, submodule in model.named_children():
            if self.can_prune(submodule):
                mask_kwargs = {'weight_mask': torch.ones_like(submodule.weight.data)}
                if isinstance(submodule, MaskedModule):
                    submodule.set_masks(**mask_kwargs)
                else:
                    masked = masked_modules[type(submodule)](submodule, **mask_kwargs)
                    new_children[name] = masked

            # Recurse for children
            self._init_mask(submodule)

        for name, masked in new_children.items():
            setattr(model, name, masked)

    def apply(self):
        self.init_mask()
        # torch.save(self.model.state_dict(), 'model.pth')
        masks = self.model_masks()
        total=0
        remain=0
        num=0
        for k,p in masks.items():
            total+=p.numel()
            remain+=p.sum().item()
            num+=1
        self.log.info('remain_ratio'+str(remain/total))
        self.log.info('pruned_layers'+str(num))

    @staticmethod
    def can_prune(module):
        if hasattr(module, 'is_classifier'):
            return not module.is_classifier
        if isinstance(module, (MaskedModule, nn.Linear, nn.Conv2d)):
            return True
        return False

    def _prunable_modules(self):
        prunable = [module for module in self.model.modules() if self.can_prune(module)]
        return prunable

    def __str__(self):
        return repr(self)

    def module_params(self, module):
        return get_params(module)

    def get_param_gradients(self):
        gradients = dict()
        for module in self._prunable_modules():
            assert module not in gradients
            if module.weight.grad is not None:
                # gradients[module] = torch.clone(module.weight.grad).detach().to(torch.device('cpu'))
                gradients[module] = (module.weight.grad)
        return gradients

    def update_gradients(self):
        self._param_gradients = self.get_param_gradients()

    def module_param_gradients(self, module):
        if not hasattr(self, "_param_gradients"):
            self.update_gradients()
        return self._param_gradients[module]




