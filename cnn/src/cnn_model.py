"""
# Script Overview
This script contains the final reusable CNN model definitions experimented on the notebooks

"""

import torch.nn as nn


def build_cnn_model():
    """Build a CNN for binary chest X-ray classification."""

    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(8192, 2)
    )

    return model