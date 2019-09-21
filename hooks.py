import numpy as np
import torch
from torch import nn
from collections import defaultdict


class Hooks:
    def __init__(self, net):
        self.hooks = []
        self.process_layers(net)
        self.num_iters = 0

        self.activations = {}
        self.gradients = {}

    def process_layers(self, module):
        if type(module) == nn.Conv2d:
            module.register_forward_hook(self.fwd)
            module.register_backward_hook(self.back)

        for c in module.children():
            self.process_layers(c)


    def fwd(self, module, inp, out):
        # go to np to make sure autograd isn't messed up here ... horribly inefficient tho
        self.activations[str(module)] = out.data.cpu().numpy()

    def back(self, module, inp, out):
        assert len(out) == 1, len(out)
        out = out[0]
        self.gradients[str(module)] = out.data.cpu().numpy()

    def show_me(self):
        print('--- ACTIVATIONS REPORT ---')
        for key, val in self.activations.items():
            print(f'{key} -> Mean = {val[-1].mean():.4f}, Std = {val[-1].std():.4f}, (found {len(val)} activations)')

        print('--- GRADIENTS REPORT ---')
        for key, val in self.gradients.items():
            print(f'{key} -> Mean = {val[-1].mean():.4f}, Std = {val[-1].std():.4f}, (found {len(val)} gradient)')
