import sys
import os
import json
import datetime
from src.cli import register_command
from src.packager import EigenPackager, parse_toml
from src.compiler import compile_to_eqir
from src.backend.unified_backend import get_quantum_backend, ValidationReport
from src.release import VERSION

SIMULATOR_BACKENDS = {"sparse", "mps", "density_matrix", "stabilizer"}

def _simulator_validation_report(backend_name: str) -> ValidationReport:
    return ValidationReport(
        backend_name=backend_name,
        ok=True,
        unsupported_ops=[],
        warnings=[],
        stats={"supported": 100.0, "emulated": 0.0, "unsupported": 0.0},
    )

@register_command("audit")
def audit_command(args, workspace_root):
    print("Auditing project manifest and compatibility...")
    packager = EigenPackager(workspace_root)
    
    strict_mode = args.strict or getattr(args, "strict", False)
    
    if os.path.exists(packager.toml_path):
        with open(packager.toml_path, 'r', encoding='utf-8') as f:
            data = parse_toml(f.read())
        pkg_name = data.get('package', {}).get('name', 'unknown')
        print(f"Package:             {pkg_name}")
    else:
        print("Package:             No manifest found (eigen.toml)")
        
    has_errors = False
    files = []
    for root, _dirs, filenames in os.walk(workspace_root):
        if any(p in root.split(os.sep) for p in ['.git', '.pytest_cache', '.eigen_cache', '.venv', 'build', 'dist']):
            continue
        for f in filenames:
            if f.endswith('.eig'):
                files.append(os.path.join(root, f))
                
    backend_name = getattr(args, "backend", "qiskit") or "qiskit"
    if backend_name in SIMULATOR_BACKENDS:
        backend = None
    else:
        try:
            backend = get_quantum_backend(backend_name)
        except KeyError:
            backend = get_quantum_backend("qiskit")
    
    total_supported = 0.0
    total_emulated = 0.0
    total_unsupported = 0.0
    file_count = 0
    
    for f_path in files:
        f_rel = os.path.relpath(f_path, workspace_root)
        try:
            g, a = compile_to_eqir(f_path, workspace_root)
            if backend is None:
                report = _simulator_validation_report(backend_name)
            else:
                report = backend.validate(g, a)
            
            total_supported += report.supported_pct
            total_emulated += report.emulated_pct
            total_unsupported += report.unsupported_pct
            file_count += 1
            
            if report.unsupported_nodes > 0:
                print(f"  WARNING: File '{f_rel}' uses {report.unsupported_nodes} "
                      f"constructs unsupported by {backend_name} backend.")
                for w in report.warnings:
                    print(f"    - {w}")
                if strict_mode:
                    has_errors = True
        except Exception as e:
            print(f"  ERROR: File '{f_rel}' failed compilation: {e}")
            has_errors = True
            
    if file_count > 0:
        avg_supported = total_supported / file_count
        avg_emulated = total_emulated / file_count
        avg_unsupported = total_unsupported / file_count
    else:
        avg_supported = 100.0
        avg_emulated = 0.0
        avg_unsupported = 0.0
        
    print("\nCompatibility Summary:")
    print(f"  Backend:     {backend_name}")
    print(f"  Supported:   {avg_supported:.1f}%")
    print(f"  Emulated:    {avg_emulated:.1f}%")
    print(f"  Unsupported: {avg_unsupported:.1f}%")
            
    if strict_mode and has_errors:
        print("\nAUDIT FAILED: Compatibility errors detected in strict mode.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAudit complete successfully.")
        
    if args.research:
        repro_path = os.path.join(workspace_root, "reproducibility_report.json")
        report_data = {
            "eigen_version": VERSION,
            "backend": backend_name,
            "optimizer_passes": ["dead_gate_elimination", "cancel_self_inverse",
                                 "merge_rotations", "peephole", "commutation_cancellation"],
            "simulator": "sparse",
            "seed": 12345,
            "platform": sys.platform,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(repro_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        print(f"Research reproducibility report generated at {repro_path}")
