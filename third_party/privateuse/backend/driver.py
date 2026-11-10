import hashlib
import tempfile
import os
import subprocess
import platform
import importlib.util
import sys

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
def _scalar_args(args):
    return tuple(arg for arg in args if not hasattr(arg, "data_ptr"))


def _cache_key(kernel_ir, launcher_src, grid, scalars, device_digest):
    digest = hashlib.sha256()
    for part in (device_digest, kernel_ir, launcher_src, repr((grid, scalars))):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def _ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def _extracted_type(ty):
    if isinstance(ty, tuple):
        val = ','.join(map(_extracted_type, ty))
        return f"[{val}]"
    if ty[0] == '*':
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
    if isinstance(ty, tuple):
        val = ''.join(map(_format_of, ty))
        return f"({val})"
    if ty[0] == '*':
        return "O"
    if ty == "constexpr":
        return "O"
    if ty.startswith("tensordesc"):
        return "O"
    return {
        "PyObject*": "O",
        "constexpr": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[_ty_to_cpp(ty)]


def _generate_cpp_launcher(constants, signature, kernel_name):
    """Launcher for the C++ (puc) kernels. Fully independent from the llir
    launcher: the kernel takes raw home-annotated pointers (`DRAM/SRAM T*`)
    instead of MLIR memref descriptors, and no ExecutionEngine/CRunnerUtils
    helpers are used. The SRAM allocation pool is reset before every launch
    (the per-kernel staging buffers are scratch memory)."""

    args_format = ''.join("O" if ty[0] == "*" else _format_of(ty) for ty in signature.values())
    format = "iiiOOOO" + args_format

    arg_decls = ', '.join(
        f"void* arg{i}" if ty[0] == "*" else f"{_ty_to_cpp(ty)} arg{i}"
        for i, ty in signature.items()
        if ty != "constexpr")
    kernel_arg_decls = arg_decls + ', ' if arg_decls else ''
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    kernel_parameters = ', '.join(
        f"arg{i}" if ty[0] == "*" else f"static_cast<{_ty_to_cpp(ty)}>(arg{i})"
        for i, ty in signature.items()
        if ty != "constexpr")
    kernel_parameters += ', ' if kernel_parameters else ''

    return f"""
#include <cstdio>
#include <cstdint>
#include <Python.h>

extern "C" {{
  void {kernel_name}({kernel_arg_decls.rstrip(', ')});
}}

static void _launch(int gridX, int gridY, int gridZ, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    {kernel_name}({kernel_parameters.rstrip(', ')});
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!ret) {{
      ptr_info.valid = false;
      return ptr_info;
    }}
    if (!PyLong_Check(ret)) {{
      Py_DECREF(ret);
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, "{format}", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items() if ty != "constexpr")});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  "__privateuse_kernel_launcher",
  NULL,
  -1,
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___privateuse_kernel_launcher(void) {{
  return PyModule_Create(&ModuleDef);
}}
"""


def compile_module(launcher_src, kernel_placeholder_name, metadata, kernel_arg_index):
    py_version = sys.version_info
    py_include_dir = os.path.join(sys.base_prefix, 'include',
                                  f'python{sys.version_info.major}.{sys.version_info.minor}')
    py_lib_dir = os.path.join(sys.base_prefix, 'lib')
    py_lib = 'python{major}.{minor}'.format(major=py_version.major, minor=py_version.minor)

    def launch(gridX, gridY, gridZ, stream, cu_function, kernel_metadata, launch_metadata,
               launch_enter_hook, launch_exit_hook, *args):
        kernel_name = kernel_metadata[6]
        kernel_args = [args[index] for index in kernel_arg_index]
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)

        key = _cache_key(cu_function, src, (gridX, gridY, gridZ), _scalar_args(args), metadata.device_digest)
        cache = get_cache_manager(key)
        name = "__privateuse_kernel_launcher"

        filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
            from triton.backends.privateuse.compiler import lower_kernel_ir
            kernel_obj = lower_kernel_ir(cu_function, (gridX, gridY, gridZ), metadata.device_info, kernel_name, kernel_args)

            with tempfile.TemporaryDirectory() as tmpdir:
                kernel_so_path = cache.put(kernel_obj, f"{kernel_name}.so", binary=True)
                launcher_src_path = os.path.join(tmpdir, "main.cpp")
                so_path = os.path.join(tmpdir, "kernel.so")
                Path(launcher_src_path).write_text(src)

                subprocess.check_call([
                    "g++", "-std=c++17", launcher_src_path, kernel_so_path,
                    f"-I{py_include_dir}", f"-L{py_lib_dir}",
                    "-shared", f"-l{py_lib}", "-fPIC", "-o", so_path
                ])

                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)

        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.launch(gridX, gridY, gridZ, kernel_metadata, launch_metadata,
                          launch_enter_hook, launch_exit_hook, *args)

    return launch


class PrivateUseLauncher(object):
    @staticmethod
    def cpu_or_self(value):
        import torch
        if not isinstance(value, torch.Tensor) or value.device.type == "cpu":
            return value
        out = torch.empty_strided(value.shape, value.stride(), device="cpu", dtype=value.dtype)
        out.copy_(value)
        return out

    @staticmethod
    def copy_or_none(value, moved):
        if value is moved:
            return
        import torch
        if isinstance(value, torch.Tensor) and isinstance(moved, torch.Tensor):
            value.copy_(moved)

    @staticmethod
    def check_devices(args, kwargs):
        import torch
        devices = {value.device for value in list(args) + list(kwargs.values()) if isinstance(value, torch.Tensor)}
        if all(device.type == "cuda" for device in devices):
            return
        raise ValueError(
            "privateuse kernels take cuda tensors only, got: " + ", ".join(sorted(str(device) for device in devices))
        )

    @staticmethod
    def launcher_wrapper(launch, cpu):
        def wrapper(*args, **kwargs):
            PrivateUseLauncher.check_devices(args, kwargs)

            if not cpu:
                return launch(*args, **kwargs)

            moved_args = [PrivateUseLauncher.cpu_or_self(arg) for arg in args]
            moved_kwargs = {key: PrivateUseLauncher.cpu_or_self(value) for key, value in kwargs.items()}

            launch(*moved_args, **moved_kwargs)

            for arg, moved in zip(args, moved_args):
                PrivateUseLauncher.copy_or_none(arg, moved)
            for key, value in kwargs.items():
                PrivateUseLauncher.copy_or_none(value, moved_kwargs[key])

        return wrapper

    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()

        def cst_key(i):
            return src.fn.arg_names.index(i) if isinstance(i, str) else i

        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        launcher_src = _generate_cpp_launcher(constants, signature, kernel_placeholder_name)

        kernel_arg_index = [index for index, ty in sorted(signature.items()) if ty != "constexpr"]
        launcher = compile_module(launcher_src, kernel_placeholder_name, metadata, kernel_arg_index)
        runtime_backend = metadata.device_info["lower"]["runtime_backend"]
        self.launch = PrivateUseLauncher.launcher_wrapper(launcher, runtime_backend == "cpu")

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class PrivateUseUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(PrivateUseUtils, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (None, kernel_obj.decode("utf-8"), None, None, sys.maxsize)


class PrivateUseDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = PrivateUseUtils()
        self.launcher_cls = PrivateUseLauncher
        self.binary_ext = "ttir"

    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_capability(self):
        return ("privateuse", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        return "privateuse"

    def set_current_device(self, device):
        assert device == "privateuse"
        return

    def get_current_target(self):
        return GPUTarget("privateuse", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
