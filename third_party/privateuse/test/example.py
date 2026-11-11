import torch

@torch.compile
def fn(x, y):
    return torch.ops.aten.floor_divide(x, y)

a = torch.randn(2, 3).cuda()
b = torch.randn(2, 3).cuda()

out = fn(a, b)
