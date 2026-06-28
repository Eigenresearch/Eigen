import os
import sys
import subprocess
import ctypes
import llvmlite.binding as llvm

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.semantic.import_resolver import ImportResolver
from src.semantic.type_checker import TypeChecker
from src.backend.ebc_compiler import EBCCompiler
from src.ir.ssa.ssa_builder import SSABuilder
from src.ir.ssa.optimizer import optimize_ebc

from src.aot.type_extractor import TypeExtractor
from src.aot.llvm_codegen import LLVMCodegen

# Initialize LLVM
try:
    llvm.initialize()
except Exception:
    pass
try:
    llvm.initialize_native_target()
except Exception:
    pass
try:
    llvm.initialize_native_asmprinter()
except Exception:
    pass

def get_qrt_lib_path(workspace_root: str) -> str:
    # Determine the static library file name
    lib_name = "eigen_native.lib" if sys.platform == "win32" else "libeigen_native.a"
    
    # Try release target first, then debug
    release_path = os.path.join(workspace_root, "native", "rust", "target", "release", lib_name)
    debug_path = os.path.join(workspace_root, "native", "rust", "target", "debug", lib_name)
    
    if os.path.exists(release_path):
        return release_path
    if os.path.exists(debug_path):
        return debug_path
        
    # If not found, attempt to build it
    rust_dir = os.path.join(workspace_root, "native", "rust")
    if os.path.exists(rust_dir):
        print("Static library not found. Compiling native runtime library (eigen_qrt)...")
        try:
            subprocess.run(["cargo", "build", "--release", "--no-default-features"], cwd=rust_dir, check=True)
            if os.path.exists(release_path):
                return release_path
        except Exception as e:
            print(f"Warning: Failed to compile native library automatically: {e}")
            
    # Try fallback in deps
    deps_release = os.path.join(workspace_root, "native", "rust", "target", "release", "deps", lib_name)
    if os.path.exists(deps_release):
        return deps_release
        
    return None

def load_native_library(workspace_root: str):
    # Prioritize searching for compiled dynamic library in workspace
    exts = [".pyd", ".dll", ".so", ".dylib"]
    
    # Try looking in native/rust/target/release and target/debug first to be fast and specific
    search_dirs = [
        os.path.join(workspace_root, "native", "rust", "target", "release"),
        os.path.join(workspace_root, "native", "rust", "target", "debug"),
    ]
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for file in os.listdir(search_dir):
                if any(file.endswith(ext) for ext in exts) and "eigen_native" in file:
                    path = os.path.join(search_dir, file)
                    try:
                        lib = ctypes.CDLL(path)
                        if hasattr(lib, "eigen_qrt_init"):
                            llvm.load_library_permanently(path)
                            return
                    except Exception:
                        pass

    # Generic workspace search
    for root, dirs, files in os.walk(os.path.join(workspace_root, "native", "rust")):
        # Skip search dirs we already checked
        if "target\\release" in root or "target/release" in root or "target\\debug" in root or "target/debug" in root:
            continue
        for file in files:
            if any(file.endswith(ext) for ext in exts) and "eigen_native" in file:
                path = os.path.join(root, file)
                try:
                    lib = ctypes.CDLL(path)
                    if hasattr(lib, "eigen_qrt_init"):
                        llvm.load_library_permanently(path)
                        return
                except Exception:
                    pass

    # Fallback to importing (e.g. from site-packages or local editable install)
    try:
        import eigen_native
        if hasattr(eigen_native, '__file__') and eigen_native.__file__:
            path = eigen_native.__file__
            if any(path.endswith(ext) for ext in exts):
                try:
                    lib = ctypes.CDLL(path)
                    if hasattr(lib, "eigen_qrt_init"):
                        llvm.load_library_permanently(path)
                        return
                except Exception:
                    pass
            else:
                dir_path = os.path.dirname(path)
                for file in os.listdir(dir_path):
                    if any(file.endswith(ext) for ext in exts) and "eigen_native" in file:
                        try:
                            lib_p = os.path.join(dir_path, file)
                            lib = ctypes.CDLL(lib_p)
                            if hasattr(lib, "eigen_qrt_init"):
                                llvm.load_library_permanently(lib_p)
                                return
                        except Exception:
                            pass
    except ImportError:
        pass

def link_object_file(obj_path: str, lib_path: str, out_path: str, lto: bool = False, strip: bool = False):
    # Find Python library path/args dynamically
    py_link_args = []
    if sys.platform == "win32":
        py_lib_dir = os.path.join(sys.base_prefix, "libs")
        py_lib_path = os.path.join(py_lib_dir, f"python3{sys.version_info.minor}.lib")
        if not os.path.exists(py_lib_path):
            py_lib_path = os.path.join(py_lib_dir, "python3.lib")
        py_link_args = [py_lib_path]
    else:
        import sysconfig
        libdir = sysconfig.get_config_var('LIBDIR')
        ldversion = sysconfig.get_config_var('LDVERSION')
        if libdir:
            py_link_args.append(f"-L{libdir}")
        if ldversion:
            py_link_args.append(f"-lpython{ldversion}")

    # Build extra flags
    extra_clang_gcc = []
    extra_msvc = []
    extra_rustc = []
    
    if lto:
        extra_clang_gcc.append("-flto")
    if strip:
        extra_clang_gcc.append("-s")
        extra_msvc.append("/OPT:REF")
        extra_rustc.extend(["-C", "strip=symbols"])

    # Try clang
    try:
        cmd = ["clang", obj_path, lib_path] + py_link_args + extra_clang_gcc + ["-o", out_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        pass

    # Try gcc
    try:
        cmd = ["gcc", obj_path, lib_path] + py_link_args + extra_clang_gcc + ["-o", out_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        pass

    # Try MSVC link (on Windows)
    if sys.platform == "win32":
        try:
            cmd = ["link", obj_path, lib_path] + py_link_args + extra_msvc + [f"/OUT:{out_path}", "ws2_32.lib", "userenv.lib", "ntdll.lib", "kernel32.lib", "msvcrt.lib", "ucrt.lib", "vcruntime.lib"]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception:
            pass

    # Try rustc fallback (especially useful on environments without clang/gcc/link on path)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".rs", delete=False, mode="w") as f:
            f.write("#![no_main]\n")
            dummy_path = f.name
        try:
            # We pass linker args through -C link-arg
            cmd = ["rustc", "--crate-type", "bin"] + extra_rustc + ["-C", f"link-arg={obj_path}", "-C", f"link-arg={lib_path}"]
            for arg in py_link_args:
                cmd.extend(["-C", f"link-arg={arg}"])
            if sys.platform == "win32":
                cmd.extend([
                    "-C", "link-arg=/ENTRY:main",
                    "-C", "link-arg=ws2_32.lib",
                    "-C", "link-arg=userenv.lib",
                    "-C", "link-arg=ntdll.lib",
                    "-C", "link-arg=kernel32.lib",
                    "-C", "link-arg=msvcrt.lib",
                    "-C", "link-arg=ucrt.lib",
                    "-C", "link-arg=vcruntime.lib"
                ])
            cmd.extend(["-o", out_path, dummy_path])
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        finally:
            try:
                os.remove(dummy_path)
            except Exception:
                pass
    except Exception:
        pass

    raise RuntimeError(
        "Linker error: Failed to link AOT binary. "
        "Please ensure a system linker (clang, gcc, or MSVC link.exe) is installed and available in PATH."
    )


class AOTCompiler:
    def __init__(self, safe_mode: bool = False):
        self.safe_mode = safe_mode

    def _compile_to_llvm_module(self, source_path: str, workspace_root: str, optimize: bool = True, seed: int = 0, emit_qir: bool = False):
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lexer = Lexer(content)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()

        resolver = ImportResolver(workspace_root)
        ast = resolver.resolve(ast)

        type_checker = TypeChecker()
        type_checker.check(ast)

        # Extract types and function signatures
        extractor = TypeExtractor()
        var_types, funcs, func_params = extractor.extract(ast)

        # Check for unsupported types (structs, maps, any) to fail fast
        allowed_types = {'int', 'float', 'bool', 'cbit', 'qubit', 'string', 'void', 'None'}
        for scope, vars in var_types.items():
            for v_name, v_type in vars.items():
                if v_type not in allowed_types:
                    if v_type in ('any', 'dynamic'):
                        raise TypeError("AOT does not support dynamic types; use --vm")
                    else:
                        raise TypeError("AOT does not support structs; use --vm")

        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        if optimize:
            instrs = optimize_ebc(instrs)

        ssa_builder = SSABuilder()
        blocks, _ = ssa_builder.build_ssa(instrs)

        # Retrieve resolved function indices from resolved_qfuncs
        resolved_qfuncs = getattr(compiler, 'resolved_qfuncs', {})
        main_start_idx = 0
        if instrs:
            # The first instruction is the JMP main_start
            if instrs[0].opcode == "JMP":
                main_start_idx = instrs[0].arg

        codegen = LLVMCodegen(var_types, funcs, func_params, safe_mode=self.safe_mode, emit_qir=emit_qir)
        llvm_module = codegen.compile_program(blocks, resolved_qfuncs, main_start_idx, seed=seed)
        return llvm_module

    def compile(self, source_path: str, workspace_root: str, optimize: bool = True, seed: int = 0,
                opt_level: int = 2, lto: bool = False, strip: bool = False, target_triple: str = None,
                emit_qir: bool = False) -> str:
        llvm_module = self._compile_to_llvm_module(source_path, workspace_root, optimize, seed, emit_qir)
        if target_triple:
            target = llvm.Target.from_triple(target_triple)
        else:
            target = llvm.Target.from_default_triple()
            
        target_machine = target.create_target_machine(codemodel='default')
        llvm_module.triple = target_machine.triple
        llvm_module.data_layout = str(target_machine.target_data)

        # Build LLVM IR string and parse it using LLVM bindings
        ir_str = str(llvm_module)
        mod_ref = llvm.parse_assembly(ir_str)

        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = opt_level
        pto.size_level = 0
        pass_builder = llvm.create_pass_builder(target_machine, pto)
        pm = pass_builder.getModulePassManager()
        pm.run(mod_ref, pass_builder)

        # Emit object file
        obj_data = target_machine.emit_object(mod_ref)
        
        # Output paths
        base_path = source_path.rsplit('.', 1)[0]
        obj_ext = ".obj" if sys.platform == "win32" else ".o"
        obj_path = base_path + obj_ext
        exe_ext = ".exe" if sys.platform == "win32" else ""
        exe_path = base_path + exe_ext

        with open(obj_path, "wb") as f:
            f.write(obj_data)

        # Get quantum static library
        lib_path = get_qrt_lib_path(workspace_root)
        if not lib_path:
            raise RuntimeError("Static library (eigen_native) not found and cargo build failed.")

        # Link
        link_object_file(obj_path, lib_path, exe_path, lto=lto, strip=strip)

        # Clean up object file
        try:
            os.remove(obj_path)
        except Exception:
            pass

        return exe_path

    def jit_execute(self, source_path: str, workspace_root: str, seed: int = 0):
        # Load the native library dynamic symbols so JIT can resolve them
        load_native_library(workspace_root)

        llvm_module = self._compile_to_llvm_module(source_path, workspace_root, optimize=True, seed=seed)
        # Target and optimizations
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine(codemodel='default')
        llvm_module.triple = target_machine.triple
        llvm_module.data_layout = str(target_machine.target_data)

        ir_str = str(llvm_module)
        mod_ref = llvm.parse_assembly(ir_str)

        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = 2
        pto.size_level = 0
        pass_builder = llvm.create_pass_builder(target_machine, pto)
        pm = pass_builder.getModulePassManager()
        pm.run(mod_ref, pass_builder)

        # Create execution engine
        engine = llvm.create_mcjit_compiler(mod_ref, target_machine)
        engine.finalize_object()

        # Get and call main function pointer
        fptr = engine.get_function_address("main")
        if not fptr:
            raise RuntimeError("Failed to resolve function address for main.")

        # Call the main() function returning i32
        ctypes.CFUNCTYPE(ctypes.c_int)(fptr)()
