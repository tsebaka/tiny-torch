import os
import torch
import sys
from torch.utils.cpp_extension import load

functional = load(
    name="functional",
    sources=["./softmax.cu"],
    verbose=True,
    build_directory="./build"
)

# after, call softmax = functional.softmax
