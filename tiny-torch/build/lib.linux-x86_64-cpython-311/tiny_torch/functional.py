import torch
import tiny_torch._C as _C

def relu(input: torch.Tensor) -> torch.Tensor:
    return _C.relu(input)
