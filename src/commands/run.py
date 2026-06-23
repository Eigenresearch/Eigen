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
from src.backend.ebc_compiler import EBCCompiler

@register_command("run")
def run_command(args, workspace_root):
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
        vm = EigenVM(trace_mode=args.trace, noise_model=noise_model, gpu_platform=gpu_platform)
        vm.execute(instructions)
        return

    strict_mode = args.strict or getattr(args, "strict", False)
    
    graph, ast = compile_to_eqir(args.file, workspace_root)
    
    if args.optimize:
        optimizer = EQIROptimizer()
        graph = optimizer.optimize(graph)
        print(f"EQIR Optimizer: Performed {optimizer.optimizations_count} optimization rewrites.")
        
    if args.backend in ("qiskit", "ibmq"):
        from src.backend.qiskit_backend import QiskitBackend
        backend_transpiler = QiskitBackend()
        qiskit_script, report = backend_transpiler.transpile(graph, ast)
        
        if strict_mode and report.unsupported_nodes > 0:
            print(f"ERROR: Backend capabilities violation for {args.backend}.", file=sys.stderr)
            for w in report.warnings:
                print(f"  - {w}", file=sys.stderr)
            sys.exit(1)
            
        out_path = args.file.rsplit('.', 1)[0] + "_qiskit.py"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(qiskit_script)
        print(f"Transpiled Qiskit script saved to '{out_path}'")
        return

    profiler = EQIRProfiler()
    profiler.start()
    
    sim_backend_type = 'dense'
    if args.backend == 'auto':
        allocated_qubits = set()
        cnot_count = 0
        gate_count = 0
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                allocated_qubits.update(node.targets)
            elif node.type == 'GATE':
                gate_count += 1
                if node.gate_name in ('CNOT', 'CZ', 'SWAP'):
                    cnot_count += 1
        num_qubits = len(allocated_qubits)
        density = gate_count / max(1, num_qubits)
        entanglement = cnot_count / max(1, num_qubits)
        
        if num_qubits <= 12:
            sim_backend_type = 'dense'
        elif entanglement < 0.25 and num_qubits > 16:
            sim_backend_type = 'mps'
        elif density < 2.0 and num_qubits > 12:
            sim_backend_type = 'sparse'
        else:
            sim_backend_type = 'dense' if num_qubits <= 16 else 'sparse'
            
        print(f"[Auto Backend] Selected '{sim_backend_type}' simulator target based on circuit metrics:")
        print(f"  Qubits:       {num_qubits}")
        print(f"  Gate Count:   {gate_count}")
        print(f"  Gate Density: {density:.2f}")
        print(f"  Entanglement: {entanglement:.2f}")
    elif args.backend == 'sparse':
        sim_backend_type = 'sparse'
    elif args.backend == 'mps':
        sim_backend_type = 'mps'
    else:
        sim_backend_type = 'dense'

    if args.vm:
        compiler = EBCCompiler()
        instructions = compiler.compile_eqir(graph)
        if args.optimize:
            from src.ir.ssa.optimizer import optimize_ebc
            instructions = optimize_ebc(instructions)
        save_to_cache(args.file, workspace_root, "ebc", instructions)
        from src.noise.noise_model import NoiseModel
        noise_model = NoiseModel(args.noise, args.noise_prob)
        gpu_platform = getattr(args, 'gpu', 'none')
        
        vm = EigenVM(trace_mode=args.trace, noise_model=noise_model, sim_type=sim_backend_type, gpu_platform=gpu_platform)
        if sim_backend_type == 'sparse':
            vm.simulator.is_sparse = True
        try:
            vm.execute(instructions)
        except AssertionError as ae:
            print(f"Assertion Error: {ae}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            write_crash_report(e, vm.call_stack, vm.ip, instructions[vm.ip].opcode if vm.ip < len(instructions) else "HALT", vm.globals)
            sys.exit(1)
    else:
        save_to_cache(args.file, workspace_root, "eqir", graph)
        from src.noise.noise_model import NoiseModel
        noise_model = NoiseModel(args.noise, args.noise_prob)
        gpu_platform = getattr(args, 'gpu', 'none')
        
        runtime = EigenRuntime(trace_mode=args.trace, noise_model=noise_model, sim_type=sim_backend_type, gpu_platform=gpu_platform)
        if sim_backend_type == 'sparse':
            runtime.simulator.is_sparse = True
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
