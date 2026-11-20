import os
import sys
import re
import subprocess
import hashlib
import functools
import tempfile

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from triton.backends.compiler import BaseBackend, GPUTarget
from triton.runtime.cache import triton_key

from triton._C.libtriton import get_cache_invalidating_env_vars, ir, passes, privateuse


_BACKEND_DIR = Path(__file__).resolve().parent
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


@functools.lru_cache(maxsize=1)
def _compiler_digest():
    digest = hashlib.sha256()
    digest.update(triton_key().encode("utf-8"))
    for path in sorted(_BACKEND_DIR.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(_BACKEND_DIR).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    digest.update(repr(sorted(get_cache_invalidating_env_vars().items())).encode("utf-8"))
    return digest.hexdigest()


def _entry_name(mod):
    _FUNC_RE = re.compile(r"^\s*(?:tt\.func|func\.func)\s+(?:(?:private|public|nested)\s+)?@([\w.$]+)", re.MULTILINE)
    found = _FUNC_RE.search(mod.str())
    if found is None:
        raise RuntimeError("privateuse: no func.func found in the module after make_ttir")
    return found.group(1)


def _check_entry_symbol(cpp, kernel_name):
    symbols = _EXTERN_C_RE.findall(cpp)
    if kernel_name + "_launcher" not in symbols:
        raise RuntimeError(
            f"privateuse: lowered kernel exports {symbols}, the launcher expects '{kernel_name}_launcher'"
        )


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
        match = re.match(
            rf"\s*(?:tt\.func|func\.func)\s+(?:(?:private|public|nested)\s+)?@{re.escape(kernel_name)}\(", line
        )
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


class LoweringPipeline:
    runtime_backend: str = "cpu"
    num_pids: int = 32
    num_blocks: int = 5
    block_bytes: int = 1 << 18
    launcher_attr: str = 'extern "C"'
    kernel_attr: str = "GLOBAL"
    outlined_attr: str = "OUTLINED"

    def digest(self):
        settings = {name: getattr(self, name) for name in type(self).__annotations__}
        return hashlib.sha256(repr((sorted(settings.items()), _compiler_digest())).encode("utf-8")).hexdigest()

    def __call__(self, kernel_ir, grid, kernel_name, args):
        constants = self._kernel_constants(kernel_ir, kernel_name, args)
        module = self._parse_ir(kernel_ir)
        module = self._specialize(module, grid, constants)
        module = self._fuse_and_bufferize(module)
        module = self._outline_computes(module)
        print(module)
        module = self._transform(module)
        module = self._lower_to_pumem(module)
        cpp = self._emit_cpp(module)
        print(cpp)
        _check_entry_symbol(cpp, kernel_name)
        return self._compile_object(cpp)

    def _kernel_constants(self, kernel_ir, kernel_name, args):
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

    def _parse_ir(self, kernel_ir):
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

    def _specialize(self, module, grid, constants):
        pm = ir.pass_manager(module.context)
        pm.enable_debug()
        privateuse.add_specialize_kernel(pm, constants, [int(dim) for dim in grid])
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        privateuse.add_lower_structured_access(pm)
        privateuse.add_lower_unstructured_access(pm)
        privateuse.add_ptr_to_memref(pm)
        passes.common.add_canonicalizer(pm)
        privateuse.add_arith_to_linalg(pm)
        privateuse.add_pid_loop(pm, self.num_pids)
        passes.common.add_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(module, 'specialize_kernel')
        return module

    def _fuse_and_bufferize(self, module):
        pm = ir.pass_manager(module.context)
        pm.enable_debug()
        privateuse.add_collapse_shape(pm)
        privateuse.add_empty_tensor_to_alloc_tensor(pm)
        privateuse.add_bufferize(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(module, 'fuse_and_bufferize')
        return module

    def _outline_computes(self, module):
        pm = ir.pass_manager(module.context)
        pm.enable_debug()
        privateuse.add_memref_to_pustream(pm, self.outlined_attr)
        pm.run(module, 'outline_computes')
        return module

    def _transform(self, module):
        pm = ir.pass_manager(module.context)
        pm.enable_debug()
        privateuse.add_lower_linalg_reduce(pm)
        privateuse.add_convert_linalg_to_loops(pm)
        passes.common.add_canonicalizer(pm)
        privateuse.add_scf_to_pumem(pm)
        privateuse.add_arith_to_puarith(pm)
        privateuse.add_math_to_pumath(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(module, 'transform')
        return module

    def _lower_to_pumem(self, module):
        pm = ir.pass_manager(module.context)
        pm.enable_debug()
        privateuse.add_expand_strided_metadata(pm)
        privateuse.add_memref_expand(pm)
        privateuse.add_lower_affine(pm)
        privateuse.add_memref_to_pumem(pm)
        privateuse.add_sink_constant_call_args(pm)
        privateuse.add_arith_to_puarith(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(module, 'lower_to_pumem')
        return module

    def _emit_cpp(self, module):
        pm = ir.pass_manager(module.context)
        pm.enable_debug()
        privateuse.add_reconcile_unrealized_casts(pm)
        privateuse.add_to_puc(pm)
        privateuse.add_puc_localize_carried(pm)
        privateuse.add_puc_simplify_assign(pm)
        privateuse.add_puc_void_ptr_args(pm)
        passes.common.add_cse(pm)
        privateuse.add_puc_launch(pm, self.launcher_attr, self.kernel_attr)
        passes.common.add_canonicalizer(pm)
        pm.run(module, 'emit_cpp')
        return privateuse.translate_puc_to_cpp(module, self.runtime_backend, self.num_pids)

    def _compile_object(self, cpp):
        runtime_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime")
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "kernel.cpp")
            so_path = os.path.join(tmpdir, "kernel.so")
            Path(src_path).write_text(cpp)
            subprocess.check_call([
                "g++", "-std=gnu++17", "-O2", "-pthread", "-fPIC", "-shared", f"-I{runtime_dir}", src_path,
                os.path.join(runtime_dir, "cpu_backend_runtime.cpp"),
                os.path.join(runtime_dir, "pumem_async_runtime.cpp"), "-o", so_path
            ])
            return Path(so_path).read_bytes()


class DeviceInfo:
    def __init__(self, fields):
        self.fields = dict(fields)

    @property
    def pipeline(self):
        return LoweringPipeline()

    @staticmethod
    def from_options(target):
        fields = {
            "target_backend": target.backend,
            "target_arch": target.arch,
            "target_warp_size": target.warp_size,
            "python_abi": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        return DeviceInfo(fields)

    @staticmethod
    def from_dict(payload):
        return DeviceInfo(payload["fields"])

    def digest(self):
        payload = repr((sorted(self.fields.items()), self.pipeline.digest()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        return mod

    def add_stages(self, stages, options, language):
        device_info = DeviceInfo.from_options(self.target)
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, device_info)

    @functools.lru_cache()
    def hash(self):
        return (self.target, _compiler_digest())

    def get_module_map(self) -> Dict[str, Any]:
        return {}
