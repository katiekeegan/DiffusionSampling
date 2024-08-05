from __future__ import print_function

import argparse
import io

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

def generate_random_means(p, n):
    means = []
    for _ in range(p):
        mean = torch.randn(n)  # Generate a random mean vector of size n
        means.append(mean)
    return means


# def generate_random_variances(p, n):
#     variances = []
#     for _ in range(p):
#         # Generate positive values for the diagonal
#         diagonal = torch.abs(torch.randn(n)) + 0.1  # Ensure positive values
#         covariance_matrix = torch.diag(diagonal)
#         variances.append(covariance_matrix)
#     return variances
def generate_random_variances(p, n):
    variances = []
    for _ in range(p):
        # Generate a random positive definite matrix for covariance
        A = torch.randn(n, n)
        covariance_matrix = torch.mm(A, A.t()) + torch.eye(n) * 0.1  # Ensure positive definiteness
        variances.append(covariance_matrix)
    return variances

def generate_random_weights(p):
    # Generate random weights that sum up to 1
    weights = np.random.dirichlet(alpha=np.ones(p), size=1).flatten()
    weights = torch.tensor(weights)
    return weights / weights.sum()  # Normalize weights to sum up to 1

def alphabar(alpha, t):
    alphabar = alpha
    for i in range(0, t - 1):
        alphabar = alphabar * alpha
    return alphabar


def sigma(alphas, alphabars, t):
    numerator = (1 - alphas[t]) * (1 - alphabars[t - 1])
    denominator = 1 - alphabars[t]
    return np.sqrt(numerator / denominator)


def generate_noising_schedule(alpha0, T, beta=0.001):
    alpha_schedule = [alpha0 - beta * i for i in range(T)]
    return alpha_schedule


def pack_params(means, variances, weights):
    # Flatten and concatenate all means
    packed_means = torch.cat([mean.reshape(-1) for mean in means])

    # Flatten and concatenate all variances
    # packed_variances = torch.cat([torch.diagonal(var).reshape(-1) for var in variances])
    packed_variances = torch.cat([var.flatten() for var in variances])

    # Convert weights to tensor if they aren't already
    weights = torch.tensor(weights) if not isinstance(weights, torch.Tensor) else weights

    # Concatenate means, variances, and weights into a single vector
    packed_params = torch.cat([packed_means, packed_variances, weights])

    return packed_params


def unpack_params(packed_params, p, n):
    """
    Unpack the single tensor back into means, variances, and weights.
    """
    means = []
    variances = []
    idx = 0

    for i in range(p):
        means.append(packed_params[idx:idx + n])
        idx += n

    # for i in range(p):
    #     diagonal = packed_params[idx:idx + n]
    #     variances.append(torch.diag(diagonal))  # Create diagonal matrix from stored diagonal elements
    #     idx += n
    for i in range(p):
        variances.append(packed_params[idx:idx + n * n].view(n, n))
        idx += n * n

    weights = packed_params[idx:idx + p]

    # Debugging statements
    print(f"Unpacked Means: {means}")
    print(f"Unpacked Variances: {variances}")
    print(f"Unpacked Weights: {weights}")

    return means, variances, weights