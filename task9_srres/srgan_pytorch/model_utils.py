import torch
from torch import nn
from torch.nn import init


def w_init(blocks, stdev=0.02):

    assert isinstance(blocks, list), 'w_init expects a list of modules'
    for block in blocks:
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                init.normal_(module.weight, std=stdev)


def g_init(blocks, mean=1.0, stdev=0.02):

    assert isinstance(blocks, list), 'g_init expects a list of modules'
    for block in blocks:
        for module in block.modules():
            if isinstance(module, nn.BatchNorm2d):
                init.normal_(module.weight, mean=mean, std=stdev)

