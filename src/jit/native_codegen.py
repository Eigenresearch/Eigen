from src.backend.bytecode import Opcode, Instruction

def generate_python_source(block: list[Instruction], vm=None) -> str:
    lines = []
    lines.append("def compiled_block(stack, locals_map, globals_map, lookup_var, vm):")
    lines.append("    pop = stack.pop")
    lines.append("    append = stack.append")
    
    # Identify loaded-only vars (hoist candidates) and loaded+stored vars (local cache candidates)
    hoist_vars = set()
    local_vars = set()
    if vm is not None:
        loaded_vars = set()
        stored_vars = set()
        for inst in block:
            if inst.opcode == Opcode.LOAD_VAR:
                loaded_vars.add(inst.arg)
            elif inst.opcode == Opcode.STORE_VAR:
                stored_vars.add(inst.arg)
            elif inst.opcode == Opcode.LOAD_CONST_STORE:
                stored_vars.add(inst.arg[1])
            elif inst.opcode in (
                Opcode.LOAD_VAR_LOAD_CONST_ADD,
                Opcode.LOAD_VAR_LOAD_CONST_SUB,
                Opcode.LOAD_VAR_LOAD_CONST_LT,
                Opcode.LOAD_VAR_LOAD_CONST_GT,
                Opcode.LOAD_VAR_LOAD_CONST_LTE,
                Opcode.LOAD_VAR_LOAD_CONST_GTE
            ):
                loaded_vars.add(inst.arg[0])
        hoist_vars = loaded_vars - stored_vars
        local_vars = loaded_vars & stored_vars
        
        # Hoist read-only variables with type guards
        for var_name in sorted(hoist_vars):
            clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
            try:
                val = vm.lookup_var(var_name)
                expected_type_name = type(val).__name__
            except Exception:
                expected_type_name = "NoneType"
                
            lines.append(f"    try:")
            lines.append(f"        var_cache_{clean_name} = lookup_var({repr(var_name)})")
            lines.append(f"    except Exception:")
            lines.append(f"        return True")
            lines.append(f"    if type(var_cache_{clean_name}).__name__ != {repr(expected_type_name)}:")
            lines.append(f"        return True")
        
        # Cache read-write variables in Python locals (avoids dict lookup per access)
        for var_name in sorted(local_vars):
            clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
            lines.append(f"    _lv_{clean_name} = locals_map.get({repr(var_name)}, 0)")
    
    # Precompute whether we're inside a function
    lines.append("    _has_frame = bool(vm.call_stack)")
    
    _var_cache = {}
    for idx, inst in enumerate(block):
        opcode = inst.opcode
        arg = inst.arg
        
        if opcode == Opcode.LOAD_CONST:
            lines.append(f"    append({repr(arg)})")
            
        elif opcode == Opcode.LOAD_VAR:
            if arg in hoist_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in arg)
                lines.append(f"    append(var_cache_{clean_name})")
            elif arg in local_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in arg)
                lines.append(f"    append(_lv_{clean_name})")
            else:
                lines.append(f"    append(lookup_var({repr(arg)}))")
            
        elif opcode == Opcode.STORE_VAR:
            if arg in local_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in arg)
                lines.append(f"    _lv_{clean_name} = pop()")
                lines.append(f"    locals_map[{repr(arg)}] = _lv_{clean_name}")
            else:
                lines.append(f"    if _has_frame:")
                lines.append(f"        vm.call_stack[-1].locals[{repr(arg)}] = pop()")
                lines.append(f"    else:")
                lines.append(f"        globals_map[{repr(arg)}] = pop()")
            
        elif opcode == Opcode.ADD:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a + b)")
            
        elif opcode == Opcode.SUB:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a - b)")
            
        elif opcode == Opcode.MUL:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a * b)")
            
        elif opcode == Opcode.DIV:
            lines.append("    b = pop(); a = pop()")
            lines.append("    if b == 0:")
            lines.append("        vm.throw_exception('DivisionByZeroError: Division by zero.')")
            lines.append("    else:")
            lines.append("        append(a / b)")
            
        elif opcode == Opcode.EQ:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a == b)")
            
        elif opcode == Opcode.NEQ:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a != b)")
            
        elif opcode == Opcode.LT:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a < b)")
            
        elif opcode == Opcode.GT:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a > b)")
            
        elif opcode == Opcode.LTE:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a <= b)")
            
        elif opcode == Opcode.GTE:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a >= b)")
            
        elif opcode == Opcode.AND:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(bool(a) and bool(b))")
            
        elif opcode == Opcode.OR:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(bool(a) or bool(b))")
            
        elif opcode == Opcode.NOT:
            lines.append("    a = pop()")
            lines.append("    append(not a)")
            
        elif opcode == Opcode.PRINT:
            lines.append("    val = pop()")
            lines.append("    if vm.output_stream is not None:")
            lines.append("        vm.output_stream.write(f'{val}\\n')")
            lines.append("    else:")
            lines.append("        if getattr(vm, 'verbose', False):")
            lines.append("            print(f'[PRINT DIRECTIVE] {val}')")
            lines.append("        else:")
            lines.append("            print(val)")
            
        elif opcode == Opcode.MOD:
            lines.append("    b = pop(); a = pop()")
            lines.append("    if b == 0:")
            lines.append("        vm.throw_exception('DivisionByZeroError: Division by zero.')")
            lines.append("    else:")
            lines.append("        append(a % b)")

        elif opcode == Opcode.POW:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a ** b)")
            
        elif opcode == Opcode.BIT_AND:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a & b)")
            
        elif opcode == Opcode.BIT_OR:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a | b)")
            
        elif opcode == Opcode.BIT_XOR:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a ^ b)")
            
        elif opcode == Opcode.BIT_NOT:
            lines.append("    a = pop()")
            lines.append("    append(~a)")
            
        elif opcode == Opcode.SHL:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a << b)")
            
        elif opcode == Opcode.SHR:
            lines.append("    b = pop(); a = pop()")
            lines.append("    append(a >> b)")
            
        elif opcode == Opcode.LOAD_CONST_STORE:
            const_val, var_name = arg
            if var_name in local_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
                lines.append(f"    _lv_{clean_name} = {repr(const_val)}")
                lines.append(f"    locals_map[{repr(var_name)}] = _lv_{clean_name}")
            else:
                lines.append(f"    if _has_frame:")
                lines.append(f"        vm.call_stack[-1].locals[{repr(var_name)}] = {repr(const_val)}")
                lines.append(f"    else:")
                lines.append(f"        globals_map[{repr(var_name)}] = {repr(const_val)}")
            
        elif opcode == Opcode.LOAD_VAR_LOAD_CONST_ADD:
            var_name, const_val = arg
            if var_name in hoist_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
                lines.append(f"    append(var_cache_{clean_name} + {repr(const_val)})")
            else:
                lines.append(f"    append(lookup_var({repr(var_name)}) + {repr(const_val)})")
            
        elif opcode == Opcode.LOAD_VAR_LOAD_CONST_SUB:
            var_name, const_val = arg
            if var_name in hoist_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
                lines.append(f"    append(var_cache_{clean_name} - {repr(const_val)})")
            else:
                lines.append(f"    append(lookup_var({repr(var_name)}) - {repr(const_val)})")
            
        elif opcode == Opcode.LOAD_VAR_LOAD_CONST_LT:
            var_name, const_val = arg
            if var_name in hoist_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
                lines.append(f"    append(var_cache_{clean_name} < {repr(const_val)})")
            else:
                lines.append(f"    append(lookup_var({repr(var_name)}) < {repr(const_val)})")
            
        elif opcode == Opcode.LOAD_VAR_LOAD_CONST_GT:
            var_name, const_val = arg
            if var_name in hoist_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
                lines.append(f"    append(var_cache_{clean_name} > {repr(const_val)})")
            else:
                lines.append(f"    append(lookup_var({repr(var_name)}) > {repr(const_val)})")
            
        elif opcode == Opcode.LOAD_VAR_LOAD_CONST_LTE:
            var_name, const_val = arg
            if var_name in hoist_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
                lines.append(f"    append(var_cache_{clean_name} <= {repr(const_val)})")
            else:
                lines.append(f"    append(lookup_var({repr(var_name)}) <= {repr(const_val)})")
            
        elif opcode == Opcode.LOAD_VAR_LOAD_CONST_GTE:
            var_name, const_val = arg
            if var_name in hoist_vars:
                clean_name = "".join(c if c.isalnum() else "_" for c in var_name)
                lines.append(f"    append(var_cache_{clean_name} >= {repr(const_val)})")
            else:
                lines.append(f"    append(lookup_var({repr(var_name)}) >= {repr(const_val)})")
            
        elif opcode == Opcode.JMP:
            lines.append(f"    vm.ip = {arg}")
            lines.append("    return True")
            
        elif opcode == Opcode.JMP_IF_FALSE:
            lines.append("    cond = pop()")
            lines.append("    if not cond:")
            lines.append(f"        vm.ip = {arg}")
            lines.append("        return True")
            
        elif opcode == Opcode.JMP_IF_TRUE:
            lines.append("    cond = pop()")
            lines.append("    if cond:")
            lines.append(f"        vm.ip = {arg}")
            lines.append("        return True")
            
        elif opcode == Opcode.HALT:
            lines.append("    vm.ip = len(vm.instructions)")
            lines.append("    return True")
            
        elif opcode == Opcode.RET:
            lines.append("    vm.op_ret(None)")
            lines.append("    return True")
            
        elif opcode == Opcode.CALL:
            lines.append(f"    vm.ip = vm.ip + {idx} + 1")
            lines.append(f"    vm.op_call({repr(arg)})")
            lines.append("    return True")
            
        else:
            lines.append(f"    vm.ip = vm.ip + {idx} + 1")
            op_name = opcode if isinstance(opcode, str) else getattr(opcode, 'name', str(opcode))
            method_name = f"op_{op_name.lower()}"
            lines.append(f"    if hasattr(vm, '{method_name}'):")
            lines.append(f"        getattr(vm, '{method_name}')({repr(arg)})")
            lines.append("    return True")
            
    lines.append(f"    vm.ip = vm.ip + {len(block)}")
    lines.append("    return False")
    return "\n".join(lines)
