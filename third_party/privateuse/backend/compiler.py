from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, privateuse
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import hashlib
import functools
import os
import re
import sys
import tempfile


_EXTERN_C_RE = re.compile(r'extern\s+"C"\s+[\w\s\*&<>:]*?\b(\w+)\s*\(')
_PTR_RE = re.compile(r"!tt\.ptr<(\w+)>")

_DTYPE_TO_IR = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "float64": "f64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "ui8",
    "uint16": "ui16",
    "uint32": "ui32",
    "uint64": "ui64",
}


def _entry_name(mod):
    _FUNC_RE = re.compile(r"^\s*(?:tt\.func|func\.func)\s+(?:(?:private|public|nested)\s+)?@([\w.$]+)", re.MULTILINE)
    found = _FUNC_RE.search(mod.str())
    if found is None:
        raise RuntimeError("privateuse: no func.func found in the module after make_ttir")
    return found.group(1)


class DeviceInfo:
    """Single source of truth for everything that shapes the generated kernel.

    A value that can change the artifact belongs here -- either as a device
    fact (fields) or as a lowering parameter (lower) -- so the cache key
    derived from it stays complete. Nothing downstream may take a codegen
    parameter from anywhere else.
    """

    def __init__(self, fields, lower):
        self.fields = dict(fields)
        self.lower = dict(lower)

    @staticmethod
    def from_options(target, options):
        fields = {
            "target_backend": target.backend,
            "target_arch": target.arch,
            "target_warp_size": target.warp_size,
            "python_abi": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        lower = {
            "runtime_backend": options.runtime_backend,
            "num_pids": options.num_pids,
            "num_blocks": options.num_blocks,
            "block_bytes": options.block_bytes,
        }
        return DeviceInfo(fields, lower)

    @staticmethod
    def from_dict(payload):
        return DeviceInfo(payload["fields"], payload["lower"])

    def digest(self):
        payload = repr((sorted(self.fields.items()), sorted(self.lower.items())))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_entry_symbol(cpp, kernel_name):
    symbols = _EXTERN_C_RE.findall(cpp)
    if kernel_name not in symbols:
        raise RuntimeError(f"privateuse: lowered kernel exports {symbols}, the launcher expects '{kernel_name}'")


def _split_params(text):
    params, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "([<{":
            depth += 1
        elif char in ")]>}":
            if depth == 0:
                params.append(text[start:index])
                return [param.strip() for param in params if param.strip()]
            depth -= 1
        elif char == "," and depth == 0:
            params.append(text[start:index])
            start = index + 1
    return []


def _ir_param_types(kernel_ir, kernel_name):
    for line in kernel_ir.splitlines():
        match = re.match(rf"\s*(?:tt\.func|func\.func)\s+(?:(?:private|public|nested)\s+)?@{re.escape(kernel_name)}\(", line)
        if match is None:
            continue
        types = []
        for param in _split_params(line[match.end():]):
            _, _, declared = param.partition(":")
            types.append(declared.strip().split()[0] if declared.strip() else "")
        return types
    raise RuntimeError(f"privateuse: no func.func @{kernel_name} in the kernel IR")


def _arg_mismatch(arg, ir_type):
    memory = _PTR_RE.match(ir_type)
    if memory is not None:
        if not hasattr(arg, "data_ptr"):
            return f"expected a tensor, got {type(arg).__name__}"
        dtype = getattr(arg, "dtype", None)
        if dtype is None:
            return None
        actual = _DTYPE_TO_IR.get(str(dtype).split(".")[-1])
        if actual is not None and actual != memory.group(1):
            return f"expected element type {memory.group(1)}, got {dtype}"
        return None
    if ir_type.startswith("f"):
        return None if isinstance(arg, float) else f"expected a float, got {type(arg).__name__}"
    if isinstance(arg, int) and not isinstance(arg, bool):
        return None
    return f"expected an int, got {type(arg).__name__}"


def _kernel_constants(kernel_ir, kernel_name, args):
    """Check the launch arguments against the kernel IR and return the (index, scalar) pairs to fold."""
    types = _ir_param_types(kernel_ir, kernel_name)
    if len(types) != len(args):
        raise RuntimeError(f"privateuse: kernel '{kernel_name}' takes {len(types)} arguments {types}, "
                           f"got {len(args)} {[type(arg).__name__ for arg in args]}")

    constants = []
    for index, (ir_type, arg) in enumerate(zip(types, args)):
        mismatch = _arg_mismatch(arg, ir_type)
        if mismatch is not None:
            raise RuntimeError(f"privateuse: argument {index} of '{kernel_name}' is {ir_type}: {mismatch}")
        if _PTR_RE.match(ir_type) is not None:
            continue
        if not isinstance(arg, (int, float)):
            raise NotImplementedError(f"privateuse: cannot fold the scalar argument {arg!r}")
        constants.append((index, arg))
    return constants


def _parse_ir(kernel_ir):
    context = ir.context()
    ir.load_dialects(context)
    privateuse.load_dialects(context)

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
        handle.write(kernel_ir)
        path = handle.name
    try:
        module = ir.parse_mlir_module(path, context)
    finally:
        os.unlink(path)
    module.context = context
    return module


def _specialize_ir(module, grid, constants):
    pm = ir.pass_manager(module.context)
    pm.enable_debug()
    privateuse.add_specialize_kernel(pm, constants, [int(dim) for dim in grid])
    passes.common.add_cse(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module, 'specialize_kernel')
    return module


def _emit_cpp(module, device_info):
    raise NotImplementedError("privateuse: no IR to C++ emission yet")


def _compile_object(cpp, device_info):
    raise NotImplementedError("privateuse: no C++ to object compilation yet")


def lower_kernel_ir(kernel_ir, grid, device_info, kernel_name, args):
    constants = _kernel_constants(kernel_ir, kernel_name, args)
    module = _parse_ir(kernel_ir)
    module = _specialize_ir(module, grid, constants)
    print(module)
    cpp = _emit_cpp(module, device_info)
    _check_entry_symbol(cpp, kernel_name)
    return _compile_object(cpp, device_info)


@dataclass(frozen=True)
class PrivateUseOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 2
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True
    fpsan_homomorphic_casts: bool = False
    instrumentation_mode: str = ""
    runtime_backend: str = "cpu"
    num_pids: int = 32
    num_blocks: int = 5
    block_bytes: int = 1 << 18

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class PrivateUseBackend(BaseBackend):
    binary_ext = 'ttir'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'privateuse'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in PrivateUseOptions.__dataclass_fields__.keys() if k in opts})
        return PrivateUseOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        return (metadata.num_warps, metadata.num_ctas, metadata.shared, metadata.cluster_dims[0],
                metadata.cluster_dims[1], metadata.cluster_dims[2], metadata.name)

    def load_dialects(self, ctx):
        privateuse.load_dialects(ctx)
        return

    @staticmethod
    def make_ttir(mod, metadata, options, device_info):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        metadata["name"] = _entry_name(mod)
        metadata["device_info"] = device_info
        metadata["device_digest"] = device_info.digest()
        print(mod)
        return mod

    def add_stages(self, stages, options, language):
        device_info = DeviceInfo.from_options(self.target, options)
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, device_info)

    @functools.lru_cache()
    def hash(self):
        return self.target

    def get_module_map(self) -> Dict[str, Any]:
        return {}
