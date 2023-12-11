"""
Filename: bilateral_filter.py
Author: João Marafuz Gaspar
Date Created: 13-Oct-2023
Description: This module provides functionalities to apply a bilateral filter on an image using depth information. 
             The bilateral filter preserves edges while smoothing similar regions based on both spatial and range kernels.
"""


import cv2
import time
import numba
import numpy as np

from tqdm import tqdm
from helpers import gauss


@numba.njit
def compute_spatial_sigma(
    depth_value: float, focus_depth: float, sigma_s: float
) -> float:
    """
    Compute the spatial sigma based on the depth difference from the focus plane.

    Args:
        depth_value (float): Depth value of the pixel.
        focus_depth (float): Depth of the focus.
        sigma_s (float): The standard deviation of the spatial kernel.

    Returns:
        float: Spatial sigma for the given depth value.
    """
    # Depth difference from the focus plane
    depth_diff = np.abs(depth_value - focus_depth) + 0.0001  # Avoid being zero

    # Adjust sigma_s based on depth difference. As difference increases, sigma increases.
    spatial_sigma = sigma_s * depth_diff
    # spatial_sigma = 3 * depth_diff # This works better

    # print(f"{self.sigma_s * (1 + depth_diff)} | {3 * depth_diff}")

    return spatial_sigma


class BilateralFilter:
    def __init__(
        self, kernel_size: int, focus_depth: float, aperture_size: float = 17.5
    ):
        """
        Initialize the BilateralFilter with the given parameters.

        Args:
            kernel_size (int): Size of the kernel.
            focus_depth (float): Depth of the focus.
            aperture_size (float): Size of the aperture. Default is 17.5.
        """
        self.kernel_size = kernel_size
        self.sigma_r = 0.8  # Tuned parameter
        self.sigma_s = self.fnumber_to_sigma(aperture_size)
        self.focus_depth = focus_depth / 255.0

    def fnumber_to_sigma(self, fnumber: float) -> float:
        """
        Convert the given f-number to its corresponding sigma value.

        Args:
            fnumber (float): F-number to be converted.

        Returns:
            float: Corresponding sigma value.
        """
        # Assuming the range of f-numbers is between 2.8 and 22
        MIN_FNUMBER = 2.8
        MAX_FNUMBER = 22.0

        # MIN_SIGMA = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8 # for f/22, using thumb rule 1
        # MIN_SIGMA = (self.kernel_size - 1) / 2 / 3.2 # for f/22, using thumb rule 2
        MIN_SIGMA = 0.0001  # for f/22
        MAX_SIGMA = 10 * ((self.kernel_size - 1) / 2 / 3.2)  # for f/2.8

        # Normalize f-number to [0, 1]
        normalized_f = (fnumber - MIN_FNUMBER) / (MAX_FNUMBER - MIN_FNUMBER)

        # Linear interpolation to get sigma_s value
        sigma_s = MIN_SIGMA + (1 - normalized_f) * (MAX_SIGMA - MIN_SIGMA)

        return sigma_s

    @staticmethod
    @numba.njit
    def filter_pixel(
        I: np.ndarray,
        D: np.ndarray,
        x: int,
        y: int,
        focus_depth: float,
        sigma_r: float,
        sigma_s: float,
        half_size: int,
    ) -> np.ndarray:
        """
        Filter a single pixel (x, y) of image I using a bilateral filter, considering
        its depth D and other parameters. This function takes into account depth
        information and can preserve edges while smoothing similar regions based on
        both spatial and range kernels.

        Args:
            I (np.ndarray): The input image, a 3D array of shape (height, width, num_channels).
            D (np.ndarray): The depth map, a 2D array of shape (height, width).
            x (int): The x-coordinate of the pixel to be filtered.
            y (int): The y-coordinate of the pixel to be filtered.
            focus_depth (float): The depth value which is considered to be in focus.
            sigma_r (float): The standard deviation of the range kernel.
            sigma_s (float): The standard deviation of the spatial kernel.
            half_size (int): Half of the kernel size (i.e., if kernel size is 7, half_size is 3).

        Returns:
            np.ndarray: The filtered pixel value, a 1D array of shape (num_channels,).
        """
        sum_weights = 0
        weighted_sum = np.zeros(3)

        # Check if the current pixel's depth is out of focus
        is_out_of_focus = np.abs(D[y, x] - focus_depth) != 0

        if not is_out_of_focus:
            return I[y, x]

        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                yi, xj = y + i, x + j

                # Check boundaries
                if yi < 0 or xj < 0 or yi >= D.shape[0] or xj >= D.shape[1]:
                    continue

                # Calculate spatial weight
                current_sigma_s = compute_spatial_sigma(
                    D[y, x], focus_depth, sigma_s
                )  # Spatially varying sigma
                g_spatial = gauss(np.sqrt(i**2 + j**2), sigma=current_sigma_s)

                # Calculate range weight
                g_range = gauss(np.abs(D[y, x] - D[yi, xj]), sigma=sigma_r)

                # Calculate bilateral weight
                w = g_spatial * g_range

                # Update weights and pixel value
                sum_weights += w
                weighted_sum += w * I[yi, xj]

        # Assign filtered pixel value to output
        return weighted_sum / sum_weights if sum_weights > 0 else I[y, x]

    def apply_filter(self, I_path: str, D_path: str, output_path: str):
        """
        Apply the bilateral filter to the image and save it to the specified output path.

        Args:
            I_path (str): The file path of the input image 'I'.
            D_path (str): The file path of the depth map 'D'.
            output_path (str): The file path where the output image 'F' will be saved.
        """
        I = cv2.imread(I_path) / 255.0  # Read image and convert to [0, 1]
        D = (
            cv2.imread(D_path, cv2.IMREAD_GRAYSCALE) / 255.0
        )  # Read depth map and convert to [0, 1]
        F = np.zeros_like(I)

        half_size = self.kernel_size // 2

        # Iterate through each pixel in the image
        start_time = time.time()
        for y in tqdm(range(F.shape[0]), desc="Filtering image"):
            for x in range(F.shape[1]):
                F[y, x] = self.filter_pixel(
                    I, D, x, y, self.focus_depth, self.sigma_r, self.sigma_s, half_size
                )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

        # Save the filtered image
        cv2.imwrite(output_path, F * 255)
