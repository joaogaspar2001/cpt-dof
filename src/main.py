"""
Computational Depth-of-Field Application
========================================

Filename: main.py
Author: João Marafuz Gaspar
Date Created: 15-Oct-2023
Description: This module initializes and runs the GUI for the Computational Depth-of-Field application,
             allowing users to interact with the application, load images, and apply filters.
"""


import sys
import argparse

from ui import UserInterface
from PyQt5.QtWidgets import QApplication


def parse_arguments():
    """
    Parses the command-line arguments provided by the user.

    Returns:
        argparse.Namespace: Parsed arguments as an object.
    """
    parser = argparse.ArgumentParser(
        description="Run the Computational Depth-of-Field application."
    )

    # Default Image
    parser.add_argument(
        "-F", "--file", help="Initial image file path to load", type=str
    )

    # CNN Depth Map Estimation
    parser.add_argument(
        "-o",
        "--optimize",
        help="Enable model optimization for CUDA when using the CNN depth map estimation - just works if CUDA exists",
        action="store_true",
    )

    # Ken Burns Effect
    parser.add_argument(
        "-w",
        "--warping-factor",
        help="Warping factor [in %%] for the Ken Burns effect",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "-n",
        "--near-plane",
        help="Near plane distance in millimeters for the Ken Burns effect",
        default=550.0,
        type=float,
    )
    parser.add_argument(
        "-f",
        "--far-plane",
        help="Far plane distance in millimeters for the Ken Burns effect",
        default=670.0,
        type=float,
    )
    parser.add_argument(
        "-d",
        "--duration-video",
        help="Duration of the video in seconds for the Ken Burns effect",
        default=5,
        type=int,
    )
    parser.add_argument(
        "-a",
        "--anaglyph",
        help="Enable anaglyph mode when using the Ken Burns effect",
        action="store_true",
    )

    return parser.parse_args()


def validate_arguments(args):
    """
    Validates the provided arguments to ensure they adhere to the required constraints.

    Args:
        args (argparse.Namespace): Parsed arguments as an object.

    Returns:
        bool: True if all arguments are valid, False otherwise.
    """
    if not (0.0 <= args.warping_factor <= 1.0):
        print("Error: Warping factor has to be in [0.0, 1.0]")
        return False

    if not (0.0 <= args.near_plane <= 590.0):
        print("Error: Near plane distance has to be in [0.0, 590.0] millimeters")
        return False

    if args.far_plane < 590.0:
        print("Error: Far plane distance has to be >= 590.0 millimeters")
        return False

    if args.duration_video < 2:
        print("Error: Duration of the video has to be >= 2 seconds")
        return False

    return True


if __name__ == "__main__":
    args = parse_arguments()

    if not validate_arguments(args):
        sys.exit(1)

    app = QApplication(sys.argv)
    ui = UserInterface(args)
    sys.exit(app.exec_())
