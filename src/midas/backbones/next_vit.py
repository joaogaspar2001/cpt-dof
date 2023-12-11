"""
Filename: next_vit.py
Added to Project: 14-Oct-2023
Description: This module contains functions and methods to utilize and potentially modify the NextViT model 
             from the timm library for specific use cases. The functions modify the forward passes and 
             potentially other mechanisms to adapt the model to particular application needs. 
             
Original Source: https://github.com/isl-org/MiDaS/blob/master/midas/backbones/next_vit.py

MIT License
Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modifications: 
- João Marafuz Gaspar: N/A
"""


import timm

import torch.nn as nn

from pathlib import Path
from .utils import activations, forward_default, get_activation

from ..external.next_vit.classification.nextvit import *


def forward_next_vit(pretrained, x):
    return forward_default(pretrained, x, "forward")


def _make_next_vit_backbone(
        model,
        hooks=[2, 6, 36, 39],
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.features[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.features[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.features[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.features[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    return pretrained


def _make_pretrained_next_vit_large_6m(hooks=None):
    model = timm.create_model("nextvit_large")

    hooks = [2, 6, 36, 39] if hooks == None else hooks
    return _make_next_vit_backbone(
        model,
        hooks=hooks,
    )
