import os
import torch
import triton
import pytest
from triton.backends.privateuse.driver import PrivateUseDriver

triton.runtime.driver.set_active(PrivateUseDriver())
device = PrivateUseDriver().get_active_torch_device()

F32_ATOL = 2e-45
F32_RTOL = 2**-23
F16_ATOL = 6e-8
F16_RTOL = 2**-11
ACC_ATOL = 1e-6
ACC_RTOL = 1e-6
SUM_ATOL = 1e-5
SUM_RTOL = 1e-5
LOWP_ATOL = 0.0
LOWP_RTOL = 2**-7
COMP_ATOL = 0.0
COMP_RTOL = 3 * 2**-23
MATH_ATOL = 0.0
MATH_RTOL = 8 * 2**-23

integerTypes = [ torch.int32 ]
floatTypes = [ torch.float32, torch.float16 ]
unsignedIntegerTypes = [ torch.uint32 ]

def make_inputs(shape, dtype, format=None):
    cpu_x = None
    if dtype in integerTypes:
        cpu_x = torch.randint(-10, 10, shape, dtype=dtype)
    if dtype in unsignedIntegerTypes:
        cpu_x = torch.randint(0, 10, shape, dtype=dtype)
    if dtype in floatTypes:
        cpu_x = torch.randn(shape, dtype=dtype)
    if format is not None:
        cpu_x = cpu_x.contiguous(memory_format=format)

    return cpu_x, cpu_x.to(device)
