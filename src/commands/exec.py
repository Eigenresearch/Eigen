import sys
import os
import json
from src.cli import register_command
from src.backend.bytecode import Instruction
from src.backend.vm import EigenVM
from src.crash_report import write_crash_report

@register_command("exec")
def exec_command(args, workspace_root):
    if not args.file.endswith('.ebc'):
        print("Error: 'eigen exec' expects a compiled EBC (.ebc) file.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(args.file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, dict):
        major = data.get("major", 1)
        if major != 3:
            print(f"UnsupportedBytecodeVersionError: Expected major version 3, got {major}.", file=sys.stderr)
            sys.exit(1)
        instructions = [Instruction.from_dict(d) for d in data["instructions"]]
    else:
        instructions = [Instruction.from_dict(d) for d in data]
        
    seed_val = getattr(args, 'seed', None)
    verbose_val = getattr(args, 'verbose', False)
    vm = EigenVM(trace_mode=args.trace, seed=seed_val, verbose=verbose_val)
    try:
        vm.execute(instructions)
    except AssertionError as ae:
        print(f"Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        write_crash_report(e, vm.call_stack, vm.ip, instructions[vm.ip].opcode if vm.ip < len(instructions) else "HALT", vm.globals)
        sys.exit(1)
