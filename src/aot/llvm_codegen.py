import llvmlite.ir as ir
from src.ir.ssa.cfg import BasicBlock

class LLVMCodegen:
    def __init__(self, var_types: dict, funcs: dict, func_params: dict,
                 safe_mode: bool = False, emit_qir: bool = False):
        self.var_types = var_types       # scope_name -> {var_name: type_str}
        self.funcs = funcs               # func_name -> (param_types, return_type)
        self.func_params = func_params   # func_name -> [param_name, ...]
        self.safe_mode = safe_mode
        self.emit_qir = emit_qir

        self.module = ir.Module(name="EigenLLVMModule")
        self.llvm_functions = {}
        self.global_vars = {}
        self.string_constants = {}

        # QIR or QRT declarations
        sim_type = ir.IntType(8).as_pointer()
        if emit_qir:
            qubit_type = ir.IntType(8).as_pointer()
            result_type = ir.IntType(8).as_pointer()
            
            # QIR opaque allocation/release
            self.qir_alloc_fn = ir.Function(
                self.module, ir.FunctionType(qubit_type, []),
                name="__quantum__rt__qubit_allocate")
            self.qir_free_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type]),
                name="__quantum__rt__qubit_release")
            
            # QIR single qubit gates
            self.qir_h_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type]),
                name="__quantum__qis__h__body")
            self.qir_x_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type]),
                name="__quantum__qis__x__body")
            self.qir_y_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type]),
                name="__quantum__qis__y__body")
            self.qir_z_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type]),
                name="__quantum__qis__z__body")
            self.qir_s_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type]),
                name="__quantum__qis__s__body")
            self.qir_t_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type]),
                name="__quantum__qis__t__body")
            
            # QIR parameterized single qubit rotation gates
            self.qir_rx_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [ir.DoubleType(), qubit_type]),
                name="__quantum__qis__rx__body")
            self.qir_ry_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [ir.DoubleType(), qubit_type]),
                name="__quantum__qis__ry__body")
            self.qir_rz_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [ir.DoubleType(), qubit_type]),
                name="__quantum__qis__rz__body")
            
            # QIR two qubit gates
            self.qir_cnot_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type, qubit_type]),
                name="__quantum__qis__cnot__body")
            self.qir_cz_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type, qubit_type]),
                name="__quantum__qis__cz__body")
            self.qir_swap_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [qubit_type, qubit_type]),
                name="__quantum__qis__swap__body")
            
            # QIR measurement and result helpers
            self.qir_measure_fn = ir.Function(
                self.module, ir.FunctionType(result_type, [qubit_type]),
                name="__quantum__qis__mz__body")
            self.qir_get_one_fn = ir.Function(
                self.module, ir.FunctionType(result_type, []),
                name="__quantum__rt__result_get_one")
            self.qir_result_equal_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.IntType(1), [result_type, result_type]),
                name="__quantum__rt__result_equal")
        else:
            self.qrt_init_fn = ir.Function(
                self.module, ir.FunctionType(sim_type, [ir.IntType(64)]),
                name="eigen_qrt_init")
            self.qrt_alloc_fn = ir.Function(
                self.module, ir.FunctionType(ir.IntType(32), [sim_type]),
                name="eigen_qrt_alloc")
            self.qrt_h_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32)]),
                name="eigen_qrt_h")
            self.qrt_x_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32)]),
                name="eigen_qrt_x")
            self.qrt_y_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32)]),
                name="eigen_qrt_y")
            self.qrt_z_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32)]),
                name="eigen_qrt_z")
            self.qrt_s_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32)]),
                name="eigen_qrt_s")
            self.qrt_t_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32)]),
                name="eigen_qrt_t")
            self.qrt_rx_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32), ir.DoubleType()]),
                name="eigen_qrt_rx")
            self.qrt_ry_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32), ir.DoubleType()]),
                name="eigen_qrt_ry")
            self.qrt_rz_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32), ir.DoubleType()]),
                name="eigen_qrt_rz")
            self.qrt_cnot_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32), ir.IntType(32)]),
                name="eigen_qrt_cnot")
            self.qrt_cz_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32), ir.IntType(32)]),
                name="eigen_qrt_cz")
            self.qrt_swap_fn = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [sim_type, ir.IntType(32), ir.IntType(32)]),
                name="eigen_qrt_swap")
            self.qrt_measure_fn = ir.Function(
                self.module, ir.FunctionType(ir.IntType(32), [sim_type, ir.IntType(32)]),
                name="eigen_qrt_measure")
            self.qrt_trace_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [sim_type]),
                name="eigen_qrt_trace")
            self.qrt_free_fn = ir.Function(
                self.module, ir.FunctionType(ir.VoidType(), [sim_type]),
                name="eigen_qrt_free")

        # Print helpers
        self.qrt_print_int_fn = ir.Function(
            self.module, ir.FunctionType(ir.VoidType(), [ir.IntType(64)]),
            name="eigen_qrt_print_int")
        self.qrt_print_bool_fn = ir.Function(
            self.module, ir.FunctionType(ir.VoidType(), [ir.IntType(1)]),
            name="eigen_qrt_print_bool")
        self.qrt_print_float_fn = ir.Function(
            self.module, ir.FunctionType(ir.VoidType(), [ir.DoubleType()]),
            name="eigen_qrt_print_float")
        self.qrt_print_string_fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]),
            name="eigen_qrt_print_string")
        
        # Panic helpers
        self.qrt_panic_div_zero_fn = ir.Function(
            self.module, ir.FunctionType(ir.VoidType(), []),
            name="eigen_qrt_panic_div_zero")

        # Trap helper
        self.trap_fn = ir.Function(self.module, ir.FunctionType(ir.VoidType(), []), name="llvm.trap")

        # Declare global simulator variable
        self.global_sim = ir.GlobalVariable(self.module, sim_type, name="global_sim")
        self.global_sim.initializer = ir.Constant(sim_type, None)

    def _get_orig_var_name(self, var_name: str) -> str:
        if not isinstance(var_name, str):
            return var_name
        
        # Handle compiler temps to avoid collision
        if var_name.startswith("_compiler_temp_"):
            parts = var_name.split('_')
            if len(parts) > 4 and parts[-1].isdigit():
                return '_'.join(parts[:-1])
            return var_name

        # Recursively strip SSA version suffixes (_\d+) from the right,
        # but stop as soon as we match a name in var_types.
        current = var_name
        while True:
            found = False
            for scope_vars in self.var_types.values():
                if current in scope_vars:
                    found = True
                    break
            if found:
                return current
            
            # Try to strip one suffix
            if '_' in current:
                parts = current.rsplit('_', 1)
                if parts[1].isdigit():
                    current = parts[0]
                    continue
            break
            
        return current

    def _get_var_type(self, f_name: str, var_name: str) -> str:
        orig_name = self._get_orig_var_name(var_name)
        t = self.var_types.get(f_name, {}).get(orig_name)
        if t is None:
            t = self.var_types.get("main", {}).get(orig_name, "int")
        return t

    def _cast_type(self, val, target_type, builder: ir.IRBuilder):
        if val.type == target_type:
            return val
        
        # Int to Float
        if isinstance(val.type, ir.IntType) and isinstance(target_type, ir.DoubleType):
            return builder.sitofp(val, ir.DoubleType())
        
        # Float to Int
        if isinstance(val.type, ir.DoubleType) and isinstance(target_type, ir.IntType):
            if target_type.width == 1:
                tmp = builder.fptosi(val, ir.IntType(64))
                return builder.trunc(tmp, ir.IntType(1))
            else:
                return builder.fptosi(val, target_type)
        
        # Int to Int
        if isinstance(val.type, ir.IntType) and isinstance(target_type, ir.IntType):
            if val.type.width > target_type.width:
                return builder.trunc(val, target_type)
            else:
                return builder.zext(val, target_type)
                
        return val

    def _map_type(self, eigen_type: str) -> ir.Type:
        if eigen_type == "int":
            return ir.IntType(64)
        elif eigen_type == "float":
            return ir.DoubleType()
        elif eigen_type in ("bool", "cbit"):
            return ir.IntType(1)
        elif eigen_type == "qubit":
            if self.emit_qir:
                return ir.IntType(8).as_pointer()
            return ir.IntType(32)
        elif eigen_type == "string":
            return ir.IntType(8).as_pointer()
        elif eigen_type in ("void", "None"):
            return ir.VoidType()
        # Default fallback
        return ir.IntType(64)

    def _get_string_constant(self, val: str, builder: ir.IRBuilder):
        if val in self.string_constants:
            return self.string_constants[val]
        str_bytes = bytearray(val.encode('utf-8')) + b'\x00'
        const_type = ir.ArrayType(ir.IntType(8), len(str_bytes))
        const_val = ir.Constant(const_type, str_bytes)
        global_var = ir.GlobalVariable(self.module, const_type, name=f".str{len(self.string_constants)}")
        global_var.initializer = const_val
        global_var.linkage = 'internal'
        global_var.global_constant = True
        ptr = builder.gep(global_var, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)], inbounds=True)
        self.string_constants[val] = ptr
        return ptr

    def compile_program(self, blocks: list[BasicBlock], resolved_qfuncs: dict,
                         main_start_idx: int, seed: int = 0) -> ir.Module:
        # Group blocks by function scope
        self.start_to_block_id = {block.start_idx: block.id for block in blocks}
        boundaries = []
        for name, idx in resolved_qfuncs.items():
            boundaries.append((idx, name))
        boundaries.append((main_start_idx, "main"))
        boundaries.sort()

        func_blocks_map = {}
        for block in blocks:
            func_name = "main"
            for i in range(len(boundaries)):
                start_idx = boundaries[i][0]
                end_idx = boundaries[i+1][0] if i + 1 < len(boundaries) else float('inf')
                if start_idx <= block.start_idx < end_idx:
                    func_name = boundaries[i][1]
                    break
            if func_name not in func_blocks_map:
                func_blocks_map[func_name] = []
            func_blocks_map[func_name].append(block)

        # Ensure "main" exists in map
        if "main" not in func_blocks_map:
            func_blocks_map["main"] = []

        all_functions = list(func_blocks_map.keys())

        # First pass: declare all LLVM functions
        for f_name in all_functions:
            if f_name == "main":
                func_type = ir.FunctionType(ir.IntType(32), [])
                llvm_func = ir.Function(self.module, func_type, name="main")
            else:
                param_types = [self._map_type(t) for t in self.funcs.get(f_name, ([], 'void'))[0]]
                ret_type = self._map_type(self.funcs.get(f_name, ([], 'void'))[1])
                func_type = ir.FunctionType(ret_type, param_types)
                llvm_func = ir.Function(self.module, func_type, name=f_name)
            self.llvm_functions[f_name] = llvm_func

        # Second pass: compile function bodies
        for f_name in all_functions:
            func_blocks = func_blocks_map[f_name]
            if not func_blocks:
                continue

            llvm_func = self.llvm_functions[f_name]
            
            entry_block = llvm_func.append_basic_block(name="entry")
            entry_builder = ir.IRBuilder(entry_block)

            # Map block ID -> LLVM Block object
            llvm_blocks = {block.id: llvm_func.append_basic_block(name=f"B{block.id}") for block in func_blocks}

            # Find all local variables in the function blocks (targets of STORE_VAR, Q_ALLOC, Q_MEASURE)
            var_names = set()
            for block in func_blocks:
                for inst in block.instructions:
                    if inst.opcode in ("STORE_VAR", "Q_ALLOC"):
                        var_names.add(self._get_orig_var_name(inst.arg))
                    elif inst.opcode == "Q_MEASURE":
                        var_names.add(self._get_orig_var_name(inst.arg[1]))

            params_list = self.func_params.get(f_name, [])
            for p_name in params_list:
                var_names.add(self._get_orig_var_name(p_name))

            # Allocate slots in the entry block
            alloca_slots = {}
            for var_name in sorted(list(var_names)):
                v_type = self._get_var_type(f_name, var_name)
                llvm_v_type = self._map_type(v_type)
                alloca_slots[var_name] = entry_builder.alloca(llvm_v_type, name=var_name)

            # If function has parameters, store incoming arguments to the alloca slots
            for i, p_name in enumerate(params_list):
                arg_val = llvm_func.args[i]
                arg_val.name = p_name
                entry_builder.store(arg_val, alloca_slots[self._get_orig_var_name(p_name)])

            # In main function, initialize simulator
            if f_name == "main":
                if not self.emit_qir:
                    seed_val = seed if seed is not None else 0
                    sim_ptr = entry_builder.call(self.qrt_init_fn, [ir.Constant(ir.IntType(64), seed_val)])
                    entry_builder.store(sim_ptr, self.global_sim)

            # Jump from entry to first real block
            first_block_llvm = llvm_blocks[func_blocks[0].id]
            entry_builder.branch(first_block_llvm)

            # Compile each block
            for block in func_blocks:
                builder = ir.IRBuilder(llvm_blocks[block.id])
                llvm_stack = []

                for inst in block.instructions:
                    op = inst.opcode
                    arg = inst.arg

                    if op == "LOAD_CONST":
                        if isinstance(arg, bool):
                            llvm_stack.append(ir.Constant(ir.IntType(1), int(arg)))
                        elif isinstance(arg, float):
                            llvm_stack.append(ir.Constant(ir.DoubleType(), arg))
                        elif isinstance(arg, int):
                            llvm_stack.append(ir.Constant(ir.IntType(64), arg))
                        elif isinstance(arg, str):
                            llvm_stack.append(self._get_string_constant(arg, builder))
                        elif arg is None:
                            llvm_stack.append(ir.Constant(ir.IntType(64), 0))

                    elif op == "LOAD_VAR":
                        orig = self._get_orig_var_name(arg)
                        if orig in alloca_slots:
                            val = builder.load(alloca_slots[orig])
                            llvm_stack.append(val)
                        else:
                            if orig not in self.global_vars:
                                v_type = self._get_var_type("main", orig)
                                llvm_v_type = self._map_type(v_type)
                                g_var = ir.GlobalVariable(self.module, llvm_v_type, name=f"g_{orig}")
                                g_var.initializer = ir.Constant(llvm_v_type, 0)
                                self.global_vars[orig] = g_var
                            g_var = self.global_vars[orig]
                            val = builder.load(g_var)
                            llvm_stack.append(val)

                    elif op == "STORE_VAR":
                        if not llvm_stack:
                            continue
                        val = llvm_stack.pop()
                        orig = self._get_orig_var_name(arg)
                        if orig in alloca_slots:
                            casted_val = self._cast_type(val, alloca_slots[orig].type.pointee, builder)
                            builder.store(casted_val, alloca_slots[orig])
                        else:
                            if orig not in self.global_vars:
                                v_type = self._get_var_type("main", orig)
                                llvm_v_type = self._map_type(v_type)
                                g_var = ir.GlobalVariable(self.module, llvm_v_type, name=f"g_{orig}")
                                g_var.initializer = ir.Constant(llvm_v_type, 0)
                                self.global_vars[orig] = g_var
                            g_var = self.global_vars[orig]
                            casted_val = self._cast_type(val, g_var.type.pointee, builder)
                            builder.store(casted_val, g_var)

                    elif op in ("ADD", "SUB", "MUL"):
                        b = llvm_stack.pop()
                        a = llvm_stack.pop()
                        if isinstance(a.type, ir.DoubleType) or isinstance(b.type, ir.DoubleType):
                            if isinstance(a.type, ir.IntType):
                                a = builder.sitofp(a, ir.DoubleType())
                            if isinstance(b.type, ir.IntType):
                                b = builder.sitofp(b, ir.DoubleType())
                            if op == "ADD":
                                res = builder.fadd(a, b)
                            elif op == "SUB":
                                res = builder.fsub(a, b)
                            elif op == "MUL":
                                res = builder.fmul(a, b)
                        else:
                            if self.safe_mode:
                                # Sadd/Ssub/Smul with overflow
                                name_map = {
                                    "ADD": "llvm.sadd.with.overflow",
                                    "SUB": "llvm.ssub.with.overflow",
                                    "MUL": "llvm.smul.with.overflow",
                                }
                                intrinsic = self.module.declare_intrinsic(name_map[op], [ir.IntType(64)])
                                ret_struct = builder.call(intrinsic, [a, b])
                                res = builder.extract_value(ret_struct, 0)
                                overflow = builder.extract_value(ret_struct, 1)
                                with builder.if_then(overflow):
                                    builder.call(self.trap_fn, [])
                            else:
                                if op == "ADD":
                                    res = builder.add(a, b)
                                elif op == "SUB":
                                    res = builder.sub(a, b)
                                elif op == "MUL":
                                    res = builder.mul(a, b)
                        llvm_stack.append(res)

                    elif op == "DIV":
                        b = llvm_stack.pop()
                        a = llvm_stack.pop()
                        if isinstance(a.type, ir.DoubleType) or isinstance(b.type, ir.DoubleType):
                            if isinstance(a.type, ir.IntType):
                                a = builder.sitofp(a, ir.DoubleType())
                            if isinstance(b.type, ir.IntType):
                                b = builder.sitofp(b, ir.DoubleType())
                            res = builder.fdiv(a, b)
                        else:
                            # 1. Division by zero check
                            is_zero = builder.icmp_signed('==', b, ir.Constant(ir.IntType(64), 0))
                            with builder.if_then(is_zero):
                                builder.call(self.qrt_panic_div_zero_fn, [])
                                builder.unreachable()

                            # 2. sdiv INT_MIN, -1 overflow check
                            int_min = ir.Constant(ir.IntType(64), -9223372036854775808)
                            minus_one = ir.Constant(ir.IntType(64), -1)

                            is_int_min = builder.icmp_signed('==', a, int_min)
                            is_minus_one = builder.icmp_signed('==', b, minus_one)
                            is_overflow = builder.and_(is_int_min, is_minus_one)

                            res_val = builder.alloca(ir.IntType(64))
                            with builder.if_else(is_overflow) as (then, otherwise):
                                with then:
                                    if self.safe_mode:
                                        builder.call(self.trap_fn, [])
                                        builder.unreachable()
                                    else:
                                        builder.store(int_min, res_val)
                                with otherwise:
                                    # Python-style floor division:
                                    q = builder.sdiv(a, b)
                                    r = builder.srem(a, b)

                                    r_not_zero = builder.icmp_signed('!=', r, ir.Constant(ir.IntType(64), 0))
                                    a_neg = builder.icmp_signed('<', a, ir.Constant(ir.IntType(64), 0))
                                    b_neg = builder.icmp_signed('<', b, ir.Constant(ir.IntType(64), 0))
                                    sign_diff = builder.xor(a_neg, b_neg)
                                    correct = builder.and_(r_not_zero, sign_diff)

                                    correction = builder.select(
                                        correct, ir.Constant(ir.IntType(64), 1),
                                        ir.Constant(ir.IntType(64), 0))
                                    q_corrected = builder.sub(q, correction)
                                    builder.store(q_corrected, res_val)
                            res = builder.load(res_val)
                        llvm_stack.append(res)

                    elif op in ("EQ", "NEQ", "LT", "GT", "LTE", "GTE"):
                        b = llvm_stack.pop()
                        a = llvm_stack.pop()
                        if isinstance(a.type, ir.DoubleType) or isinstance(b.type, ir.DoubleType):
                            if isinstance(a.type, ir.IntType):
                                a = builder.sitofp(a, ir.DoubleType())
                            if isinstance(b.type, ir.IntType):
                                b = builder.sitofp(b, ir.DoubleType())
                            cond_map = {"EQ": "oeq", "NEQ": "one", "LT": "olt", "GT": "ogt", "LTE": "ole", "GTE": "oge"}
                            res = builder.fcmp_ordered(cond_map[op], a, b)
                        else:
                            if isinstance(a.type, ir.IntType) and isinstance(b.type, ir.IntType):
                                if a.type.width != b.type.width:
                                    if a.type.width < b.type.width:
                                        a = self._cast_type(a, b.type, builder)
                                    else:
                                        b = self._cast_type(b, a.type, builder)
                            cond_map = {"EQ": "==", "NEQ": "!=", "LT": "<", "GT": ">", "LTE": "<=", "GTE": ">="}
                            res = builder.icmp_signed(cond_map[op], a, b)
                        llvm_stack.append(res)

                    elif op == "NOT":
                        a = llvm_stack.pop()
                        res = builder.xor(a, ir.Constant(ir.IntType(1), 1))
                        llvm_stack.append(res)

                    elif op in ("AND", "OR"):
                        b = llvm_stack.pop()
                        a = llvm_stack.pop()
                        if op == "AND":
                            res = builder.and_(a, b)
                        else:
                            res = builder.or_(a, b)
                        llvm_stack.append(res)

                    elif op == "JMP":
                        builder.branch(llvm_blocks[self.start_to_block_id[arg]])

                    elif op == "JMP_IF_FALSE":
                        cond = llvm_stack.pop()
                        false_block = llvm_blocks[self.start_to_block_id[arg]]
                        true_block = llvm_blocks[self.start_to_block_id[block.end_idx]]
                        builder.cbranch(cond, true_block, false_block)

                    elif op == "JMP_IF_TRUE":
                        cond = llvm_stack.pop()
                        true_block = llvm_blocks[self.start_to_block_id[arg]]
                        false_block = llvm_blocks[self.start_to_block_id[block.end_idx]]
                        builder.cbranch(cond, true_block, false_block)

                    elif op == "CALL":
                        target_idx, callee_name, num_args = arg
                        args = []
                        for _ in range(num_args):
                            args.append(llvm_stack.pop())
                        args.reverse()

                        callee_fn = self.llvm_functions[callee_name]
                        # Cast arguments if they mismatch the expected parameter types
                        casted_args = []
                        for i, actual_arg in enumerate(args):
                            expected_type = callee_fn.function_type.args[i]
                            casted_args.append(self._cast_type(actual_arg, expected_type, builder))

                        res = builder.call(callee_fn, casted_args)
                        if not isinstance(callee_fn.function_type.return_type, ir.VoidType):
                            llvm_stack.append(res)

                    elif op == "RET":
                        if f_name == "main":
                            # Free simulator
                            sim = builder.load(self.global_sim)
                            # builder.call(self.qrt_free_fn, [sim])
                            builder.ret(ir.Constant(ir.IntType(32), 0))
                        else:
                            if isinstance(llvm_func.function_type.return_type, ir.VoidType):
                                if llvm_stack:
                                    llvm_stack.pop()
                                builder.ret_void()
                            else:
                                if llvm_stack:
                                    val = llvm_stack.pop()
                                    casted_val = self._cast_type(val, llvm_func.function_type.return_type, builder)
                                    builder.ret(casted_val)
                                else:
                                    builder.ret(ir.Constant(llvm_func.function_type.return_type, 0))

                    elif op == "HALT":
                        if not self.emit_qir:
                            sim = builder.load(self.global_sim)
                            # builder.call(self.qrt_free_fn, [sim])
                        builder.ret(ir.Constant(ir.IntType(32), 0))

                    elif op == "Q_ALLOC":
                        orig = self._get_orig_var_name(arg)
                        if self.emit_qir:
                            qubit_id = builder.call(self.qir_alloc_fn, [])
                            alloc_type = ir.IntType(8).as_pointer()
                        else:
                            sim = builder.load(self.global_sim)
                            qubit_id = builder.call(self.qrt_alloc_fn, [sim])
                            alloc_type = ir.IntType(32)

                        if orig in alloca_slots:
                            casted = self._cast_type(qubit_id, alloca_slots[orig].type.pointee, builder)
                            builder.store(casted, alloca_slots[orig])
                        else:
                            if orig not in self.global_vars:
                                self.global_vars[orig] = ir.GlobalVariable(self.module, alloc_type, name=f"g_{orig}")
                                if self.emit_qir:
                                    self.global_vars[orig].initializer = ir.Constant(alloc_type, None)
                                else:
                                    self.global_vars[orig].initializer = ir.Constant(alloc_type, 0)
                            g_var = self.global_vars[orig]
                            casted = self._cast_type(qubit_id, g_var.type.pointee, builder)
                            builder.store(casted, g_var)

                    elif op == "Q_GATE":
                        gate_name, targets = arg
                        gate_upper = gate_name.upper()

                        if self.emit_qir:
                            q_ids = []
                            for t in targets:
                                orig_t = self._get_orig_var_name(t)
                                if orig_t in alloca_slots:
                                    q_ids.append(builder.load(alloca_slots[orig_t]))
                                else:
                                    q_ids.append(builder.load(self.global_vars[orig_t]))

                            if gate_upper in ('RX', 'RY', 'RZ'):
                                theta = llvm_stack.pop()
                                if gate_upper == 'RX':
                                    builder.call(self.qir_rx_fn, [theta, q_ids[0]])
                                elif gate_upper == 'RY':
                                    builder.call(self.qir_ry_fn, [theta, q_ids[0]])
                                elif gate_upper == 'RZ':
                                    builder.call(self.qir_rz_fn, [theta, q_ids[0]])
                            else:
                                if gate_upper == 'H':
                                    builder.call(self.qir_h_fn, [q_ids[0]])
                                elif gate_upper == 'X':
                                    builder.call(self.qir_x_fn, [q_ids[0]])
                                elif gate_upper == 'Y':
                                    builder.call(self.qir_y_fn, [q_ids[0]])
                                elif gate_upper == 'Z':
                                    builder.call(self.qir_z_fn, [q_ids[0]])
                                elif gate_upper == 'S':
                                    builder.call(self.qir_s_fn, [q_ids[0]])
                                elif gate_upper == 'T':
                                    builder.call(self.qir_t_fn, [q_ids[0]])
                                elif gate_upper == 'CNOT':
                                    builder.call(self.qir_cnot_fn, [q_ids[0], q_ids[1]])
                                elif gate_upper == 'CZ':
                                    builder.call(self.qir_cz_fn, [q_ids[0], q_ids[1]])
                                elif gate_upper == 'SWAP':
                                    builder.call(self.qir_swap_fn, [q_ids[0], q_ids[1]])
                        else:
                            sim = builder.load(self.global_sim)
                            if gate_upper in ('RX', 'RY', 'RZ'):
                                theta = llvm_stack.pop()
                                orig_t = self._get_orig_var_name(targets[0])
                                if orig_t in alloca_slots:
                                    q_id = builder.load(alloca_slots[orig_t])
                                else:
                                    q_id = builder.load(self.global_vars[orig_t])

                                if gate_upper == 'RX':
                                    builder.call(self.qrt_rx_fn, [sim, q_id, theta])
                                elif gate_upper == 'RY':
                                    builder.call(self.qrt_ry_fn, [sim, q_id, theta])
                                elif gate_upper == 'RZ':
                                    builder.call(self.qrt_rz_fn, [sim, q_id, theta])
                            else:
                                q_ids = []
                                for t in targets:
                                    orig_t = self._get_orig_var_name(t)
                                    if orig_t in alloca_slots:
                                        q_ids.append(builder.load(alloca_slots[orig_t]))
                                    else:
                                        q_ids.append(builder.load(self.global_vars[orig_t]))

                                if gate_upper == 'H':
                                    builder.call(self.qrt_h_fn, [sim, q_ids[0]])
                                elif gate_upper == 'X':
                                    builder.call(self.qrt_x_fn, [sim, q_ids[0]])
                                elif gate_upper == 'Y':
                                    builder.call(self.qrt_y_fn, [sim, q_ids[0]])
                                elif gate_upper == 'Z':
                                    builder.call(self.qrt_z_fn, [sim, q_ids[0]])
                                elif gate_upper == 'S':
                                    builder.call(self.qrt_s_fn, [sim, q_ids[0]])
                                elif gate_upper == 'T':
                                    builder.call(self.qrt_t_fn, [sim, q_ids[0]])
                                elif gate_upper == 'CNOT':
                                    builder.call(self.qrt_cnot_fn, [sim, q_ids[0], q_ids[1]])
                                elif gate_upper == 'CZ':
                                    builder.call(self.qrt_cz_fn, [sim, q_ids[0], q_ids[1]])
                                elif gate_upper == 'SWAP':
                                    builder.call(self.qrt_swap_fn, [sim, q_ids[0], q_ids[1]])

                    elif op == "Q_MEASURE":
                        qubit_name, cbit_name = arg
                        orig_q = self._get_orig_var_name(qubit_name)
                        orig_c = self._get_orig_var_name(cbit_name)
                        if orig_q in alloca_slots:
                            q_id = builder.load(alloca_slots[orig_q])
                        else:
                            q_id = builder.load(self.global_vars[orig_q])

                        if self.emit_qir:
                            res_ptr = builder.call(self.qir_measure_fn, [q_id])
                            one_ptr = builder.call(self.qir_get_one_fn, [])
                            res_bool = builder.call(self.qir_result_equal_fn, [res_ptr, one_ptr])
                        else:
                            sim = builder.load(self.global_sim)
                            res_val = builder.call(self.qrt_measure_fn, [sim, q_id])
                            res_bool = builder.trunc(res_val, ir.IntType(1))

                        if orig_c in alloca_slots:
                            casted = self._cast_type(res_bool, alloca_slots[orig_c].type.pointee, builder)
                            builder.store(casted, alloca_slots[orig_c])
                        else:
                            if orig_c not in self.global_vars:
                                self.global_vars[orig_c] = ir.GlobalVariable(
                    self.module, ir.IntType(1), name=f"g_{orig_c}")
                                self.global_vars[orig_c].initializer = ir.Constant(ir.IntType(1), 0)
                            g_var = self.global_vars[orig_c]
                            casted = self._cast_type(res_bool, g_var.type.pointee, builder)
                            builder.store(casted, g_var)

                    elif op == "Q_TRACE":
                        if not self.emit_qir:
                            sim = builder.load(self.global_sim)
                            builder.call(self.qrt_trace_fn, [sim])

                    elif op == "PRINT":
                        val = llvm_stack.pop()
                        if isinstance(val.type, ir.IntType):
                            if val.type.width == 1:
                                builder.call(self.qrt_print_bool_fn, [val])
                            else:
                                builder.call(self.qrt_print_int_fn, [val])
                        elif isinstance(val.type, ir.DoubleType):
                            builder.call(self.qrt_print_float_fn, [val])
                        elif (isinstance(val.type, ir.PointerType)
                              and isinstance(val.type.pointee, ir.IntType)
                              and val.type.pointee.width == 8):
                            builder.call(self.qrt_print_string_fn, [val])

                if not builder.block.is_terminated:
                    if block.end_idx in self.start_to_block_id:
                        next_block_id = self.start_to_block_id[block.end_idx]
                        builder.branch(llvm_blocks[next_block_id])
                    else:
                        if f_name == "main":
                            sim = builder.load(self.global_sim)
                            # builder.call(self.qrt_free_fn, [sim])
                            builder.ret(ir.Constant(ir.IntType(32), 0))
                        else:
                            ret_type = llvm_func.function_type.return_type
                            if isinstance(ret_type, ir.VoidType):
                                builder.ret_void()
                            else:
                                builder.ret(ir.Constant(ret_type, 0))

        return self.module
