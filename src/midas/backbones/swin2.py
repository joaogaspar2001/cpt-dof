"""
Filename: swin2.py
Added to Project: 14-Oct-2023
Description: This module contains functions to utilize and potentially modify the Swin Transformer v2 model 
             with specific configurations and hooks. The functions are designed to work with a pretrained 
             Swin Transformer v2 model from the timm library and adjust its forward pass or other functionalities 
             for specific use cases.
             
Original Source: https://github.com/isl-org/MiDaS/blob/master/midas/backbones/swin2.py

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

from .swin_common import _make_swin_backbone


def _make_pretrained_swin2l24_384(pretrained, hooks=None):
    model = timm.create_model("swinv2_large_window12to24_192to384_22kft1k", pretrained=pretrained)

    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )


def _make_pretrained_swin2b24_384(pretrained, hooks=None):
    model = timm.create_model("swinv2_base_window12to24_192to384_22kft1k", pretrained=pretrained)

    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )


def _make_pretrained_swin2t16_256(pretrained, hooks=None):
    model = timm.create_model("swinv2_tiny_window16_256", pretrained=pretrained)

    hooks = [1, 1, 5, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks,
        patch_grid=[64, 64]
    )
