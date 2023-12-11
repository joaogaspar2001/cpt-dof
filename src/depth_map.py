"""
Filename: depth_map.py
Author: João Marafuz Gaspar
Date Created: 12-Oct-2023
Description: This module provides functionalities to generate a depth map from an image and user scribbles. 
             The depth map generation is based on a combination of the image's intensity values and user scribbles.
"""


import cv2
import time
import numba
import numpy as np

from tqdm import tqdm


class DepthMapGenerator:
    def __init__(self, beta: float = 100.0, max_iter: int = 7500):
        """
        Initialize the DepthMapGenerator with specified parameters.

        Args:
            beta (float): A parameter to adjust the weight computation's sensitivity to pixel intensity differences.
                          Higher values of beta will make the weighting more sensitive to intensity differences.
                          Defaults to 100.0.
            max_iter (int): The maximum number of iterations to perform when creating the depth map.
                            A higher value will potentially lead to a more accurate depth map but will also increase
                            computational time. Defaults to 7500.
        """
        self.beta = beta
        self.max_iter = max_iter

    @staticmethod
    @numba.njit
    def compute_weights(I: np.ndarray, beta: float) -> np.ndarray:
        """
        Compute the weights matrix based on the input image I and parameter beta.

        Args:
            I (np.ndarray): Input image matrix.
            beta (float): Parameter controlling sensitivity to intensity differences.

        Returns:
            np.ndarray: Weights matrix of shape (h, w, 4) storing the computed weights.
        """
        h, w = I.shape
        weights = np.zeros((h, w, 4), dtype=I.dtype)
        for i in range(h):
            for j in range(w):
                # Assume zero padding
                weights[i, j, 0] = np.exp(
                    -beta * np.abs(I[i, j] - (I[i - 1, j] if i > 0 else 0))
                )  # Up
                weights[i, j, 1] = np.exp(
                    -beta * np.abs(I[i, j] - (I[i + 1, j] if i < h - 1 else 0))
                )  # Down
                weights[i, j, 2] = np.exp(
                    -beta * np.abs(I[i, j] - (I[i, j - 1] if j > 0 else 0))
                )  # Left
                weights[i, j, 3] = np.exp(
                    -beta * np.abs(I[i, j] - (I[i, j + 1] if j < w - 1 else 0))
                )  # Right

        return weights

    @staticmethod
    @numba.njit
    def update_D(S: np.ndarray, D: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Update depth map D based on scribbles S and weights.

        Args:
            S (np.ndarray): Scribble matrix where -1 indicates no scribble and other values indicate depth scribbles.
            D (np.ndarray): Initial depth map.
            weights (np.ndarray): Weight matrix derived from the input image I to guide depth map generation.

        Returns:
            np.ndarray: Updated depth map in [0, 1].
        """
        h, w = S.shape
        D_next = D.copy()
        for i in range(h):
            for j in range(w):
                if S[i, j] == -1:  # If pixel is not a scribble
                    D_up = (
                        D[i - 1, j] if i > 0 else D[i, j]
                    )  # If pixel is at the top edge, use the current pixel value
                    w_up = (
                        0 if D_up == -1 else weights[i, j, 0]
                    )  # If pixel has no depth value, we can't use it

                    D_down = (
                        D[i + 1, j] if i < h - 1 else D[i, j]
                    )  # If pixel is at the bottom edge, use the current pixel value
                    w_down = (
                        0 if D_down == -1 else weights[i, j, 1]
                    )  # If pixel has no depth value, we can't use it

                    D_left = (
                        D[i, j - 1] if j > 0 else D[i, j]
                    )  # If pixel is at the left edge, use the current pixel value
                    w_left = (
                        0 if D_left == -1 else weights[i, j, 2]
                    )  # If pixel has no depth value, we can't use it

                    D_right = (
                        D[i, j + 1] if j < w - 1 else D[i, j]
                    )  # If pixel is at the right edge, use the current pixel value
                    w_right = (
                        0 if D_right == -1 else weights[i, j, 3]
                    )  # If pixel has no depth value, we can't use it

                    if (
                        w_up + w_down + w_left + w_right > 0
                    ):  # Update depth map if at least one neighbour has a depth value
                        D_next[i, j] = (
                            w_up * D_up
                            + w_down * D_down
                            + w_left * D_left
                            + w_right * D_right
                        ) / (w_up + w_down + w_left + w_right)

                else:  # If pixel is a scribble keep it as it is
                    D_next[i, j] = S[i, j]

        return D_next

    def create_depth_map(self, I_path: str, S_path: str, output_path: str):
        """
        Create a depth map given the image, scribbles, and save it to the specified output path.

        Args:
            I_path (str): Path to the input image file.
            S_path (str): Path to the scribbles image file.
            output_path (str): Path to save the generated depth map.
        """
        I = (
            cv2.imread(I_path, cv2.IMREAD_GRAYSCALE) / 255.0
        )  # Read image in greyscale and convert to [0, 1]
        S = cv2.imread(
            S_path, cv2.IMREAD_UNCHANGED
        )  # Read scribbles image in RGBA (A for representing transparency)
        S = np.where(
            S[:, :, 3] == 0, -1, S[:, :, 0] / 255.0
        )  # Replace transparent pixels with -1 and convert valid scribbles to greyscale in [0, 1]

        start_time = time.time()
        D = S.copy()  # Initialize depth map with scribbles
        weights = self.compute_weights(I, self.beta)
        for _ in tqdm(
            range(self.max_iter),
            desc="Creating Depth Map by combining user scribbles with image/predictions",
        ):
            D_next = self.update_D(S, D, weights)
            D = D_next
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

        cv2.imwrite(output_path, D * 255.0)
