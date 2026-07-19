import sys
import json
from src.cli import register_command
from src.compiler import compile_to_eqir, save_to_cache
from src.ir.optimizer import EQIROptimizer
from src.backend.bytecode import Instruction
from src.backend.vm import EigenVM
from src.runtime import EigenRuntime
from src.profiler import EQIRProfiler
from src.crash_report import write_crash_report

@register_command("run")
def run_command(args, workspace_root):
    # §6.2 (Security): Subprocess isolation for untrusted code.
    # If --sandbox is passed and we're not already in a sandbox, respawn
    # ourselves in a subprocess.
    import os
    if getattr(args, "sandbox", False) and os.environ.get("EIGEN_SANDBOX") != "1":
        import subprocess
        new_env = os.environ.copy()
        new_env["EIGEN_SANDBOX"] = "1"
        cmd = [sys.executable] + sys.argv
        # Remove --sandbox from the args to avoid recursion, although the env
        # var also guards against it.
        if "--sandbox" in cmd:
            cmd.remove("--sandbox")
        try:
            return subprocess.call(cmd, env=new_env)
        except Exception as e:
            print(f"Error launching sandbox subprocess: {e}", file=sys.stderr)
            sys.exit(1)

    opt_level = 0
    if getattr(args, "O", None) is not None:
        opt_level = args.O
    elif getattr(args, "optimize", False):
        opt_level = 2
    optimize = (opt_level >= 1)

    if getattr(args, 'aot', False):
        from src.aot.compiler import AOTCompiler
        seed_val = getattr(args, 'seed', 0)
        if seed_val is None:
            seed_val = 0
        aot = AOTCompiler()
        try:
            aot.jit_execute(args.file, workspace_root, seed=seed_val)
            return
        except Exception as e:
            print(f"WARNING: AOT JIT failed ({e}), falling back to VM", file=sys.stderr)
            args.vm = True

    if args.file.endswith('.ebc'):
        print(f"Executing EBC bytecode file '{args.file}' on VM...")
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            instructions = [Instruction.from_dict(d) for d in data["instructions"]]
        else:
            instructions = [Instruction.from_dict(d) for d in data]
        from src.noise.noise_model import NoiseModel
        noise_model = NoiseModel(args.noise, args.noise_prob)
        gpu_platform = getattr(args, 'gpu', 'none')
        seed_val = getattr(args, 'seed', None)
        verbose_val = getattr(args, 'verbose', False)
        deterministic_val = bool(getattr(args, 'deterministic', False))
        max_instr_val = getattr(args, 'max_instructions', None)
        timeout_val = getattr(args, 'instruction_timeout', None)
        if deterministic_val and seed_val is None:
            seed_val = 0
        vm = EigenVM(trace_mode=args.trace, noise_model=noise_model,
                     gpu_platform=gpu_platform, seed=seed_val,
                     verbose=verbose_val, opt_level=opt_level,
                     deterministic=deterministic_val,
                     max_instruction_count=max_instr_val,
                     instruction_timeout_s=timeout_val)
        vm.execute(instructions)
        return

    strict_mode = getattr(args, "strict", False)
    
    graph, ast = compile_to_eqir(args.file, workspace_root)
    
    if optimize:
        optimizer = EQIROptimizer()
        graph = optimizer.optimize(graph)
        print(f"EQIR Optimizer: Performed {optimizer.optimizations_count} optimization rewrites.")
        
    if args.backend in ("qiskit", "ibmq"):
        # §4.1 Unified Backend Interface: route through the QuantumBackend
        # adapter so callers don't import QiskitBackend directly.
        from src.backend.unified_backend import get_quantum_backend
        backend = get_quantum_backend(args.backend)
        report = backend.validate(graph, ast)

        if strict_mode and not report.ok:
            print(f"ERROR: Backend capabilities violation for {args.backend}.", file=sys.stderr)
            for w in report.warnings:
                print(f"  - {w}", file=sys.stderr)
            sys.exit(1)

        qiskit_script = backend.compile(graph, ast)
        out_path = args.file.rsplit('.', 1)[0] + "_qiskit.py"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(qiskit_script)
        print(f"Transpiled Qiskit script saved to '{out_path}'")
        return

    profiler = EQIRProfiler()
    profiler.start()
    
    sim_backend_type = 'dense'
    if args.backend == 'auto':
        from src.backend.gate_registry import CLIFFORD_GATES
        from src.backend.sim_selector import select_from_counts
        allocated_qubits = set()
        cnot_count = 0
        gate_count = 0
        is_all_clifford = True
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                allocated_qubits.update(node.targets)
            elif node.type == 'GATE':
                gate_count += 1
                if node.gate_name in ('CNOT', 'CZ', 'SWAP'):
                    cnot_count += 1
                if is_all_clifford and node.gate_name not in CLIFFORD_GATES:
                    is_all_clifford = False
        num_qubits = len(allocated_qubits)
        density = gate_count / max(1, num_qubits)
        entanglement = cnot_count / max(1, num_qubits)

        noise_active = bool(getattr(args, 'noise', None) is not None
                            and getattr(args, 'noise_prob', 0.0) > 0)
        report = select_from_counts(
            n_qubits=num_qubits,
            n_2q_gates=cnot_count,
            n_gates=gate_count,
            is_all_clifford=is_all_clifford,
            noise_active=noise_active,
        )
        sim_backend_type = report.chosen

        print(f"[Auto Backend] Selected '{sim_backend_type}' simulator target ({report.reason}):")
        print(f"  Qubits:       {num_qubits}")
        print(f"  Gate Count:   {gate_count}")
        print(f"  Gate Density: {density:.2f}")
        print(f"  Entanglement: {entanglement:.2f}")
        print(f"  All-Clifford: {is_all_clifford}")
        if report.fallback_used:
            print(f"  Fallback:     '{report.fallback_from}' -> '{report.chosen}' (memory budget)")
    elif args.backend == 'sparse':
        sim_backend_type = 'sparse'
    elif args.backend == 'mps':
        sim_backend_type = 'mps'
    elif args.backend == 'density_matrix':
        sim_backend_type = 'density_matrix'
    elif args.backend == 'stabilizer':
        sim_backend_type = 'stabilizer'
    else:
        sim_backend_type = 'dense'

    if args.vm:
        from src.compiler import to_ebc
        instructions = to_ebc(args.file, workspace_root, optimize=optimize)
        from src.noise.noise_model import NoiseModel
        noise_model = NoiseModel(args.noise, args.noise_prob)
        gpu_platform = getattr(args, 'gpu', 'none')
        seed_val = getattr(args, 'seed', None)
        verbose_val = getattr(args, 'verbose', False)
        deterministic_val = bool(getattr(args, 'deterministic', False))
        max_instr_val = getattr(args, 'max_instructions', None)
        timeout_val = getattr(args, 'instruction_timeout', None)
        if deterministic_val and seed_val is None:
            seed_val = 0

        vm = EigenVM(trace_mode=args.trace, noise_model=noise_model,
                     sim_type=sim_backend_type, gpu_platform=gpu_platform,
                     seed=seed_val, verbose=verbose_val,
                     opt_level=opt_level, deterministic=deterministic_val,
                     max_instruction_count=max_instr_val,
                     instruction_timeout_s=timeout_val)
        # Native-Python recursion fast path: pre-compile qualifying pure
        # recursive functions to Python callables that bypass VM dispatch.
        try:
            from src.jit.recursive_codegen import compile_recursive_functions
            vm.recursive_funcs = compile_recursive_functions(ast)
        except Exception:
            vm.recursive_funcs = {}
        try:
            vm.execute(instructions)
        except AssertionError as ae:
            print(f"Assertion Error: {ae}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            write_crash_report(e, vm.call_stack, vm.ip,
                               instructions[vm.ip].opcode if vm.ip < len(instructions) else "HALT",
                               vm.globals)
            sys.exit(1)
    else:
        save_to_cache(args.file, workspace_root, "eqir", graph)
        from src.noise.noise_model import NoiseModel
        noise_model = NoiseModel(args.noise, args.noise_prob)
        gpu_platform = getattr(args, 'gpu', 'none')
        seed_val = getattr(args, 'seed', None)
        
        runtime = EigenRuntime(trace_mode=args.trace, noise_model=noise_model,
                               sim_type=sim_backend_type, gpu_platform=gpu_platform,
                               seed=seed_val)
        try:
            runtime.execute(graph)
        except AssertionError as ae:
            print(f"Assertion Error: {ae}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Runtime Error: {e}", file=sys.stderr)
            sys.exit(1)
        
    profiler.stop()
    stats = profiler.profile(graph)
    profiler.print_profile_report(stats)
