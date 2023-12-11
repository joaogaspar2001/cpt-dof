"""
Filename: ken_burns.py
Author: João Marafuz Gaspar
Date Created: 18-Oct-2023
Description: A class to generate videos with the Ken Burns effect from input images and frames.
"""


import cv2
import time
import numpy as np

from tqdm import tqdm
from numba import jit
from helpers import gauss
from PyQt5.QtCore import QRect


def normalized_depth_to_disparity(
    depth: np.ndarray,
    iod_mm: float = 64.0,
    px_size_mm: float = 0.25,
    screen_distance_mm: float = 590.0,
    near_plane_mm: float = 550.0,
    far_plane_mm: float = 670.0,
) -> np.ndarray:
    """
    Convert normalized depth to disparity in pixels.

    Args:
        depth (np.ndarray): Normalized depth map.
        iod_mm (float): Interocular distance in millimeters. Defaults to 64.0.
        px_size_mm (float): Pixel size in millimeters. Defaults to 0.25.
        screen_distance_mm (float): Screen distance in millimeters. Defaults to 590.0.
        near_plane_mm (float): Near plane distance in millimeters. Defaults to 550.0.
        far_plane_mm (float): Far plane distance in millimeters. Defaults to 670.0.

    Returns:
        np.ndarray: Disparity map.
    """
    height, width = depth.shape
    px_disparity = np.zeros_like(depth, dtype=float)

    for y in range(height):
        for x in range(width):
            # Convert normalized depth to actual depth in mm
            actual_depth_mm = near_plane_mm + depth[y, x] * (
                far_plane_mm - near_plane_mm
            )

            # Calculate disparity in mm
            disparity_mm = (
                iod_mm * (actual_depth_mm - screen_distance_mm)
            ) / actual_depth_mm

            # Convert disparity from mm to pixels
            px_disparity[y, x] = disparity_mm / px_size_mm

    return px_disparity


@jit(nopython=True)
def forward_warp_image_jit(
    src_image: np.ndarray,
    src_depth: np.ndarray,
    disparity: np.ndarray,
    warp_factor: float,
    dst_image: np.ndarray,
    dst_mask: np.ndarray,
    dst_depth: np.ndarray,
):
    """
    Forward warp the source image using the provided disparity and warp factor.

    Args:
        src_image (np.ndarray): The source image to be warped.
        src_depth (np.ndarray): Depth information for the source image.
        disparity (np.ndarray): Disparity information used for the warping.
        warp_factor (float): The factor controlling the intensity of the warping effect.
        dst_image (np.ndarray): The resulting warped image.
        dst_mask (np.ndarray): Mask indicating valid regions after warping.
        dst_depth (np.ndarray): Depth information after warping.
    """
    height, width = src_image.shape[:2]
    for y in range(height):
        for x in range(width):
            # Compute where current pixel should be warped to
            new_x = int(round(x + disparity[y, x] * warp_factor))
            new_y = y

            if 0 <= new_x < width:
                # Perform the Z-Test
                if src_depth[y, x] < dst_depth[new_y, new_x]:
                    dst_image[new_y, new_x] = src_image[y, x]
                    dst_depth[new_y, new_x] = src_depth[y, x]
                    dst_mask[new_y, new_x] = 1


def forward_warp_image(
    src_image: np.ndarray,
    src_depth: np.ndarray,
    disparity: np.ndarray,
    warp_factor: float,
) -> (np.ndarray, np.ndarray):
    """
    Perform forward warping on the source image using disparity.

    Args:
        src_image (np.ndarray): Input source image.
        src_depth (np.ndarray): Depth map corresponding to the source image.
        disparity (np.ndarray): Disparity map for warping.
        warp_factor (float): Factor to scale the warping.

    Returns:
        (np.ndarray, np.ndarray): Warped image and its corresponding mask.
    """
    assert src_image.shape[:2] == disparity.shape
    assert src_image.shape[:2] == src_depth.shape

    dst_image = np.zeros_like(src_image)
    dst_mask = np.zeros_like(src_depth)
    dst_depth = np.ones_like(src_depth) * np.inf

    forward_warp_image_jit(
        src_image, src_depth, disparity, warp_factor, dst_image, dst_mask, dst_depth
    )

    return dst_image, dst_mask


@jit(nopython=True)
def inpaint_holes_jit(
    img: np.ndarray,
    mask: np.ndarray,
    size: int,
    sigma: float,
    result: np.ndarray,
):
    """
    Inpaint the holes in the image using a Gaussian kernel.

    Args:
        img (np.ndarray): Input image.
        mask (np.ndarray): Mask indicating areas to be inpainted.
        size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for the Gaussian.
        result (np.ndarray): Output inpainted image.
    """
    half_size = size // 2
    height, width = img.shape[:2]

    for y in range(height):
        for x in range(width):
            if mask[y, x] < 0.5:
                r_sum, g_sum, b_sum, w_sum = 0, 0, 0, 0
                for i in range(-half_size, half_size + 1):
                    for j in range(-half_size, half_size + 1):
                        xi, yj = x + i, y + j

                        if 0 <= xi < width and 0 <= yj < height:
                            if mask[yj, xi] >= 0.5:
                                dist = np.sqrt(i**2 + j**2)
                                w = gauss(dist, sigma)

                                r_sum += w * img[yj, xi, 0]
                                g_sum += w * img[yj, xi, 1]
                                b_sum += w * img[yj, xi, 2]
                                w_sum += w

                if w_sum > 0:
                    result[y, x, 0] = r_sum / w_sum
                    result[y, x, 1] = g_sum / w_sum
                    result[y, x, 2] = b_sum / w_sum


def inpaint_holes(img: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    """
    Inpaint holes in the image using a Gaussian kernel.

    Args:
        img (np.ndarray): Input image.
        mask (np.ndarray): Mask indicating areas to be inpainted.
        size (int): Size of the Gaussian kernel.

    Returns:
        np.ndarray: Inpainted image.
    """
    sigma = (size - 1) / 2 / 3.2
    result = np.copy(img)
    inpaint_holes_jit(img, mask, size, sigma, result)
    return result


@jit(nopython=True)
def create_anaglyph(
    image_left: np.ndarray, image_right: np.ndarray, saturation: float
) -> np.ndarray:
    """
    Create an anaglyph image from left and right images with applied saturation.

    Args:
        image_left (np.ndarray): The left image represented as a numpy array.
        image_right (np.ndarray): The right image represented as a numpy array.
        saturation (float): The saturation scale to be applied.

    Returns:
        np.ndarray: The anaglyph image represented as a numpy array.
    """
    height, width, _ = image_left.shape
    anaglyph = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            pixel_left_bgr = image_left[y, x]
            pixel_right_bgr = image_right[y, x]

            # Convert BGR to RGB and scale to [0, 1]
            pixel_left_rgb = pixel_left_bgr[::-1] / 255.0
            pixel_right_rgb = pixel_right_bgr[::-1] / 255.0

            # Convert RGB to HSV
            V_left = max(pixel_left_rgb)
            m_left = min(pixel_left_rgb)
            C_left = V_left - m_left

            if C_left != 0:
                S_left = C_left / V_left
            else:
                S_left = 0

            if C_left == 0:
                H_left = 0
            elif V_left == pixel_left_rgb[0]:
                H_left = ((pixel_left_rgb[1] - pixel_left_rgb[2]) / C_left) % 6
            elif V_left == pixel_left_rgb[1]:
                H_left = ((pixel_left_rgb[2] - pixel_left_rgb[0]) / C_left) + 2
            else:
                H_left = ((pixel_left_rgb[0] - pixel_left_rgb[1]) / C_left) + 4

            H_left /= 6

            # Scale the saturation
            S_left *= saturation

            # Convert back to RGB
            C_left = S_left * V_left
            X_left = C_left * (1 - abs((H_left * 6) % 2 - 1))
            m_left = V_left - C_left

            if 0 <= H_left < 1 / 6:
                rgb_left = [C_left, X_left, 0]
            elif 1 / 6 <= H_left < 2 / 6:
                rgb_left = [X_left, C_left, 0]
            elif 2 / 6 <= H_left < 3 / 6:
                rgb_left = [0, C_left, X_left]
            elif 3 / 6 <= H_left < 4 / 6:
                rgb_left = [0, X_left, C_left]
            elif 4 / 6 <= H_left < 5 / 6:
                rgb_left = [X_left, 0, C_left]
            else:
                rgb_left = [C_left, 0, X_left]

            rgb_left = [(color + m_left) * 255.0 for color in rgb_left]

            # Combine the two images as per instructions and convert RGB to BGR
            anaglyph[y, x] = [
                rgb_left[0],
                pixel_right_rgb[1] * 255.0,
                pixel_right_rgb[2] * 255.0,
            ][::-1]

    # Returns a single anaglyph image.
    return anaglyph


class KenBurnsEffect:
    def __init__(
        self,
        inputFilePath: str,
        depthMapFilePath: str,
        outputFilePath: str,
        frames,
        warping_factor: float = 0.5,
        near_plane_mm: float = 550.0,
        far_plane_mm: float = 670.0,
        video_duration: int = 5,
        get_anaglyph: bool = False,
    ):
        """
        Initialize the KenBurnsEffect class with input image path, depth map path,
        output path, start and end frames, and number of frames for the animation.

        Args:
            inputFilePath (str): Path to the input image.
            depthMapFilePath (str): Path to the depth map of the image.
            outputFilePath (str): Path to save the generated video.
            frames (list[QRect]): A list containing start and end QRect frames for the effect.
            warping_factor (float): Warping factor [in %] for the Ken Burns effect. Defaults to 0.5.
            near_plane_mm (float): Near plane distance in millimeters for the Ken Burns effect. Defaults to 550.0.
            far_plane_mm (float): Far plane distance in millimeters for the Ken Burns effect. Defaults to 670.0.
            video_duration (int): Animation's duration. Defaults to 5 seconds (150 frames).
            get_anaglyph (bool): Whether to generate an anaglyph video or not. Defaults to False.
        """
        self.inputFilePath = inputFilePath
        self.depthMapFilePath = depthMapFilePath
        self.outputFilePath = outputFilePath
        self.start_frame = frames[0]
        self.end_frame = frames[1]
        self.N = video_duration * 30

        self.warping_mult_const = warping_factor * 20
        self.near_plane_mm = near_plane_mm
        self.far_plane_mm = far_plane_mm
        self.get_anglyph = get_anaglyph

    @staticmethod
    def ken_burns_effect(
        image: np.ndarray,
        depth_map,
        start_frame: QRect,
        end_frame: QRect,
        N: int,
        warping_mult_const: float,
        near_plane_mm: float,
        far_plane_mm: float,
        get_anaglyph: bool,
    ) -> list[np.ndarray]:
        """
        Generates the Ken Burns effect on a given image using start and end frames.

        Args:
            image (ndarray): Input image as a NumPy array.
            start_frame (QRect): The starting frame for the effect.
            end_frame (QRect): The ending frame for the effect.
            N (int): Number of frames for the animation.
            warping_mult_const (float): Warping multiplicative constant adapted for the desired effect.
            near_plane_mm (float): Near plane distance in millimeters.
            far_plane_mm (float): Far plane distance in millimeters.
            get_anaglyph (bool): Whether to generate an anaglyph video or not.

        Returns:
            list[ndarray]: A list containing the frames with the Ken Burns effect.
        """
        depth_map_normalized = (depth_map - np.min(depth_map)) / (
            np.max(depth_map) - np.min(depth_map)
        )
        disparity = normalized_depth_to_disparity(
            depth_map_normalized, near_plane_mm=near_plane_mm, far_plane_mm=far_plane_mm
        )

        startX, startY, startWidth, startHeight = (
            start_frame.x(),
            start_frame.y(),
            start_frame.width(),
            start_frame.height(),
        )
        endX, endY, endWidth, endHeight = (
            end_frame.x(),
            end_frame.y(),
            end_frame.width(),
            end_frame.height(),
        )

        output_height, output_width = image.shape[:2]
        output_frames = []

        for i in tqdm(range(N), desc="Generating frames", ncols=100):
            x = startX + (endX - startX) * (i / (N - 1))
            y = startY + (endY - startY) * (i / (N - 1))
            width = startWidth + (endWidth - startWidth) * (i / (N - 1))
            height = startHeight + (endHeight - startHeight) * (i / (N - 1))

            image_cropped = image[int(y) : int(y + height), int(x) : int(x + width)]
            depth_map_cropped = depth_map[
                int(y) : int(y + height), int(x) : int(x + width)
            ]
            disparity_cropped = disparity[
                int(y) : int(y + height), int(x) : int(x + width)
            ]

            x_center_of_frame = x + width / 2
            normalized_x_center_of_frame = x_center_of_frame / output_width

            if get_anaglyph == False:
                warp_factor = warping_mult_const * (normalized_x_center_of_frame - 0.5)

                warped_image, mask = forward_warp_image(
                    image_cropped,
                    depth_map_cropped,
                    disparity_cropped,
                    warp_factor=warp_factor,
                )
                inpainted_image = inpaint_holes(warped_image, mask, size=19)
                resized = cv2.resize(inpainted_image, (output_width, output_height))
                output_frames.append(resized)
            else:
                warp_factor_left = warping_mult_const * (
                    normalized_x_center_of_frame - 0.5
                )
                warp_factor_right = -warp_factor_left

                warped_image_left, mask_left = forward_warp_image(
                    image_cropped,
                    depth_map_cropped,
                    disparity_cropped,
                    warp_factor=warp_factor_left,
                )
                warped_image_right, mask_right = forward_warp_image(
                    image_cropped,
                    depth_map_cropped,
                    disparity_cropped,
                    warp_factor=warp_factor_right,
                )
                inpainted_image_left = inpaint_holes(
                    warped_image_left, mask_left, size=19
                )
                inpainted_image_right = inpaint_holes(
                    warped_image_right, mask_right, size=19
                )
                anaglyph = create_anaglyph(
                    inpainted_image_left, inpainted_image_right, 0.3
                )
                resized_analgyph = cv2.resize(anaglyph, (output_width, output_height))
                output_frames.append(resized_analgyph)

        return output_frames

    def generate_video(self):
        """
        Generates a video with the Ken Burns effect applied to the input image.
        """
        image = cv2.imread(self.inputFilePath)
        depth_map = cv2.imread(self.depthMapFilePath, cv2.IMREAD_GRAYSCALE)

        start_time = time.time()
        frames = KenBurnsEffect.ken_burns_effect(
            image,
            depth_map,
            self.start_frame,
            self.end_frame,
            self.N,
            self.warping_mult_const,
            self.near_plane_mm,
            self.far_plane_mm,
            self.get_anglyph,
        )

        output_height, output_width = image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # if mp4v coded does not exist

        # Produce the video with 30 fps frame rate
        out = cv2.VideoWriter(
            self.outputFilePath, fourcc, 30.0, (output_width, output_height)
        )
        for frame in tqdm(frames, desc="Writing video", ncols=100):
            out.write(frame)
        out.release()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
