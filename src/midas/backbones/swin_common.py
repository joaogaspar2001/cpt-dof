"""
Filename: swin_common.py
Added to Project: 14-Oct-2023
Description: This module contains functions and methods to utilize and potentially modify the Swin Transformer model 
             for specific use cases. The functions modify the forward passes and potentially other mechanisms to 
             adapt the model to particular application needs. 
             
Original Source: https://github.com/isl-org/MiDaS/blob/master/midas/backbones/swin_common.py

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


import torch

import torch.nn as nn
import numpy as np

from .utils import activations, forward_default, get_activation, Transpose


def forward_swin(pretrained, x):
    return forward_default(pretrained, x)


def _make_swin_backbone(
        model,
        hooks=[1, 1, 17, 1],
        patch_grid=[96, 96]
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.layers[0].blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.layers[1].blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.layers[2].blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.layers[3].blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if hasattr(model, "patch_grid"):
        used_patch_grid = model.patch_grid
    else:
        used_patch_grid = patch_grid

    patch_grid_size = np.array(used_patch_grid, dtype=int)

    pretrained.act_postprocess1 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size(patch_grid_size.tolist()))
    )
    pretrained.act_postprocess2 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 2).tolist()))
    )
    pretrained.act_postprocess3 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 4).tolist()))
    )
    pretrained.act_postprocess4 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 8).tolist()))
    )

    return pretrained
