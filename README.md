# tiny-torch
ахуеете, работает так же быстро как у торча (а иногда даже быстрее), однако код понятный
## Install from PyPi
```sh
pip install tiny-torch-nevmenko
```
## Usage
```python
import torch
import tiny_torch.functional as F

x = torch.tensor([-1.0, 0.0, 1.0], device="cuda")
print(F.relu(x))
# tensor([0., 0., 1.], device='cuda:0')
```


<div style="text-align: center;">
  <img src="/assets/tiny-torch.png" width="200" style="display: block; margin: 0 auto;" />
</div>

Here I'll reproduce some torch functions to have a sdome experience with torch_cpp_extentions and CUDA
