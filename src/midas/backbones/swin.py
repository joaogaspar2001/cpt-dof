"""
Filename: swin.py
Added to Project: 14-Oct-2023
Description: This module contains functions to utilize and potentially modify the Swin Transformer model 
             with specific configurations and hooks. The functions are designed to work with a pretrained 
             Swin Transformer model from the timm library and adjust its forward pass or other functionalities 
             for specific use cases.
             
Original Source: https://github.com/isl-org/MiDaS/blob/master/midas/backbones/swin.py

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


def _make_pretrained_swinl12_384(pretrained, hooks=None):
    model = timm.create_model("swin_large_patch4_window12_384", pretrained=pretrained)

    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )
