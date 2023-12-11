"""
Filename: helpers.py
Author: João Marafuz Gaspar
Date Created: 12-Oct-2023
Description: This module contains helper functions that are used throughout the project. 
"""


import math
import numba


@numba.njit
def gauss(x, sigma: float = 1.0, mu: float = 0.0) -> float:
    """
    Computes the unnormalized probability density function (pdf) of a Gaussian distribution
    for a given input, standard deviation (sigma), and mean (mu).

    Args:
        x (float): The point at which to sample the Gaussian distribution.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 1.0.
        mu (float, optional): The mean of the Gaussian distribution. Defaults to 0.0.

    Returns:
        float: The unnormalized pdf value of the Gaussian distribution at the given point `x`.
    """
    exponent = (x - mu) / sigma
    return math.exp(-0.5 * exponent**2)
