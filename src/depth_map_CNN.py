"""
Filename: depth_map_CNN.py
Added to Project: 14-Oct-2023
Description: This module provides functionalities to generate a depth map from an image using a CNN-based model.

Original Source: https://github.com/isl-org/MiDaS/blob/master/run.py

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
- João Marafuz Gaspar: Adapted the code to be used as a module in the project.
"""


import cv2
import time
import torch
import warnings

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
)
import numpy as np

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


class DepthMapCNNGenerator:
    def __init__(self, optimize_execution: bool):
        """
        Initialize the DepthMapCNNGenerator.

        Args:
            optimize_execution (bool): Flag to enable model optimization for CUDA.
        """
        self.first_execution = True
        self.optimize_execution = optimize_execution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path: str, optimize: bool) -> (DPTDepthModel, Compose):
        """
        Load the depth prediction model and necessary transformations.

        Args:
            model_path (str): Path to the pre-trained model weights.
            optimize (bool): Flag to enable model optimization for CUDA.

        Returns:
            DPTDepthModel: Loaded depth prediction model.
            Compose: Image transformation pipeline.
        """
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2t16_256",
            non_negative=True,
        )
        net_w, net_h = 256, 256

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()

        if optimize and self.device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

        model.to(self.device)

        return model, transform

    def generate_depth_from_image(
        self,
        model: DPTDepthModel,
        image: np.ndarray,
        target_size: tuple,
        optimize: bool,
    ) -> np.ndarray:
        """
        Process the image using the model to generate depth map.

        Args:
            model (DPTDepthModel): Depth prediction model.
            image (np.ndarray): Image array for depth prediction.
            target_size (tuple): Target size for the depth output.
            optimize (bool): Flag to enable model optimization for CUDA.

        Returns:
            np.ndarray: Predicted depth array.
        """
        sample = torch.from_numpy(image).to(self.device).unsqueeze(0)

        if optimize and self.device == torch.device("cuda"):
            if self.first_execution:
                print("Optimization warning...")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=target_size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return prediction

    def create_depth_map_CNN(self, image_file_path: str, output_file_path: str):
        """
        Create depth maps using a CNN-based model.

        Args:
            image_file_path (str): Path to the image file.
            output_file_path (str): Path to save the depth map.
        """
        start_time = time.time()
        print("Initialize CNN prediction...")
        print("Device in usage: %s" % self.device)

        # Load depth prediction model and transformations
        model, transform = self.load_model(
            "weights/dpt_swin2_tiny_256.pt", self.optimize_execution
        )

        print("Started generated depth map from CNN...")

        # Load and normalize the original image
        original_image_rgb = (
            cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB) / 255.0
        )
        transformed_image = transform({"image": original_image_rgb})["image"]

        # Generate depth prediction
        with torch.no_grad():
            depth_prediction = self.generate_depth_from_image(
                model,
                transformed_image,
                original_image_rgb.shape[1::-1],
                self.optimize_execution,
            )

        depth_range = depth_prediction.max() - depth_prediction.min()
        out = (
            255.0
            * (depth_prediction - depth_prediction.min())
            / max(depth_range, np.finfo("float").eps)
        )
        out = 255.0 - (255.0 * (out - out.min()) / max(out.max() - out.min(), 1))
        cv2.imwrite(output_file_path, out.astype("uint8"))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished -> Elapsed Time: {elapsed_time:.2f} seconds")
