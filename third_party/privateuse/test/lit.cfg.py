import glob
import os

import lit.formats
import lit.llvm
from lit.llvm.subst import ToolSubst

lit.llvm.initialize(lit_config, config)
llvm_config = lit.llvm.llvm_config

config.name = 'PRIVATEUSE'
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root
config.environment['FILECHECK_OPTS'] = '--enable-var-scope'

triton_root = os.path.abspath(os.path.join(config.test_source_root, '..', '..', '..'))
candidates = []
for build_dir in glob.glob(os.path.join(triton_root, 'build', 'cmake.*')):
    opt_dir = os.path.join(build_dir, 'third_party', 'privateuse', 'tools', 'privateuse-opt')
    if os.path.exists(os.path.join(opt_dir, 'privateuse-opt')):
        candidates.append((os.path.getmtime(opt_dir), opt_dir))
if not candidates:
    raise RuntimeError('privateuse-opt not found; build triton first')
config.privateuse_tools_dir = max(candidates)[1]
llvm_bin_dir = os.path.abspath(os.path.join(triton_root, '..', 'llvm-project', 'build', 'bin'))

tool_dirs = [d for d in (config.privateuse_tools_dir, llvm_bin_dir) if os.path.isdir(d)]
for d in tool_dirs:
    llvm_config.with_environment('PATH', d, append_path=True)
llvm_config.add_tool_substitutions([ToolSubst('privateuse-opt'), ToolSubst('FileCheck')], tool_dirs)
