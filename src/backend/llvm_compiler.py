import os
from src.backend.bytecode import Opcode, Instruction
from src.ir.ssa.cfg import BasicBlock

class LLVMCompiler:
    def __init__(self):
        self.temp_counter = 0

    def get_temp_reg(self) -> str:
        self.temp_counter += 1
        return f"t{self.temp_counter}"

    def infer_types(self, blocks: list[BasicBlock]) -> dict:
        var_types = {}
        for block in blocks:
            stack_types = []
            for inst in block.instructions:
                opcode = inst.opcode
                arg = inst.arg
                
                if opcode == Opcode.LOAD_CONST:
                    if isinstance(arg, bool):
                        stack_types.append('bool')
                    elif isinstance(arg, float):
                        stack_types.append('float')
                    elif isinstance(arg, int):
                        stack_types.append('int')
                    else:
                        stack_types.append('int')
                elif opcode == Opcode.LOAD_VAR:
                    t = var_types.get(arg, 'int')
                    stack_types.append(t)
                elif opcode == Opcode.STORE_VAR:
                    if stack_types:
                        t = stack_types.pop()
                        var_types[arg] = t
                elif opcode in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV):
                    if len(stack_types) >= 2:
                        t2 = stack_types.pop()
                        t1 = stack_types.pop()
                        if t1 == 'float' or t2 == 'float':
                            stack_types.append('float')
                        else:
                            stack_types.append('int')
                    else:
                        stack_types.append('int')
                elif opcode in (Opcode.EQ, Opcode.NEQ, Opcode.LT, Opcode.GT, Opcode.LTE, Opcode.GTE, Opcode.AND, Opcode.OR, Opcode.NOT):
                    if opcode in (Opcode.AND, Opcode.OR):
                        if len(stack_types) >= 2:
                            stack_types.pop()
                            stack_types.pop()
                    elif opcode == Opcode.NOT:
                        if stack_types:
                            stack_types.pop()
                    else:
                        if len(stack_types) >= 2:
                            stack_types.pop()
                            stack_types.pop()
                    stack_types.append('bool')
                elif opcode == Opcode.Q_MEASURE:
                    cbit = arg[1]
                    var_types[cbit] = 'bool'
        return var_types

    def compile_ssa(self, blocks: list[BasicBlock]) -> str:
        var_types = self.infer_types(blocks)
        type_map = {
            'int': 'i32',
            'float': 'double',
            'bool': 'i1'
        }
        
        lines = [
            "; ModuleID = 'EigenLLVMModule'",
            "source_filename = \"eigen_source.eig\"",
            "",
            "%Qubit = type opaque",
            "%Result = type opaque",
            "",
            "define void @main() #0 {",
            "entry:",
            "  br label %B0",
            ""
        ]
        
        for block in blocks:
            lines.append(f"B{block.id}:")
            
            # Compile Phi nodes
            for var, phi in sorted(block.phi_nodes.items()):
                t = var_types.get(var, 'int')
                llvm_type = type_map.get(t, 'i32')
                
                phi_args = []
                for pred_id, val in phi:
                    # check if val is constant
                    if isinstance(val, (int, float, bool)):
                        if isinstance(val, bool):
                            phi_args.append(f"[ {1 if val else 0}, %B{pred_id} ]")
                        else:
                            phi_args.append(f"[ {val}, %B{pred_id} ]")
                    else:
                        phi_args.append(f"[ %{val}, %B{pred_id} ]")
                
                phi_str = ", ".join(phi_args)
                lines.append(f"  %{var} = phi {llvm_type} {phi_str}")
            
            llvm_stack = []
            
            for inst in block.instructions:
                opcode = inst.opcode
                arg = inst.arg
                
                if opcode == Opcode.LOAD_CONST:
                    if isinstance(arg, bool):
                        llvm_stack.append(f"i1 {1 if arg else 0}")
                    elif isinstance(arg, float):
                        llvm_stack.append(f"double {arg}")
                    elif isinstance(arg, int):
                        llvm_stack.append(f"i32 {arg}")
                    else:
                        llvm_stack.append(f"i32 {arg}")
                        
                elif opcode == Opcode.LOAD_VAR:
                    t = var_types.get(arg, 'int')
                    llvm_type = type_map.get(t, 'i32')
                    llvm_stack.append(f"{llvm_type} %{arg}")
                    
                elif opcode == Opcode.STORE_VAR:
                    if llvm_stack:
                        val_op = llvm_stack.pop()
                        lines.append(f"  %{arg} = select i1 true, {val_op}, {val_op}")
                        
                elif opcode in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV):
                    if len(llvm_stack) >= 2:
                        val_op2 = llvm_stack.pop()
                        val_op1 = llvm_stack.pop()
                        
                        t1, val1 = val_op1.split(' ', 1)
                        t2, val2 = val_op2.split(' ', 1)
                        
                        # Promote if necessary
                        if t1 == 'double' or t2 == 'double':
                            res_type = 'double'
                            if t1 != 'double':
                                temp = self.get_temp_reg()
                                lines.append(f"  %{temp} = sitofp {t1} {val1} to double")
                                val1 = f"%{temp}"
                            if t2 != 'double':
                                temp = self.get_temp_reg()
                                lines.append(f"  %{temp} = sitofp {t2} {val2} to double")
                                val2 = f"%{temp}"
                                
                            op_map = {
                                Opcode.ADD: 'fadd',
                                Opcode.SUB: 'fsub',
                                Opcode.MUL: 'fmul',
                                Opcode.DIV: 'fdiv'
                            }
                            op_name = op_map[opcode]
                        else:
                            res_type = 'i32'
                            op_map = {
                                Opcode.ADD: 'add',
                                Opcode.SUB: 'sub',
                                Opcode.MUL: 'mul',
                                Opcode.DIV: 'sdiv'
                            }
                            op_name = op_map[opcode]
                            
                        temp_res = self.get_temp_reg()
                        lines.append(f"  %{temp_res} = {op_name} {res_type} {val1}, {val2}")
                        llvm_stack.append(f"{res_type} %{temp_res}")
                        
                elif opcode in (Opcode.EQ, Opcode.NEQ, Opcode.LT, Opcode.GT, Opcode.LTE, Opcode.GTE):
                    if len(llvm_stack) >= 2:
                        val_op2 = llvm_stack.pop()
                        val_op1 = llvm_stack.pop()
                        
                        t1, val1 = val_op1.split(' ', 1)
                        t2, val2 = val_op2.split(' ', 1)
                        
                        if t1 == 'double' or t2 == 'double':
                            if t1 != 'double':
                                temp = self.get_temp_reg()
                                lines.append(f"  %{temp} = sitofp {t1} {val1} to double")
                                val1 = f"%{temp}"
                            if t2 != 'double':
                                temp = self.get_temp_reg()
                                lines.append(f"  %{temp} = sitofp {t2} {val2} to double")
                                val2 = f"%{temp}"
                                
                            cond_map = {
                                Opcode.EQ: 'oeq',
                                Opcode.NEQ: 'one',
                                Opcode.LT: 'olt',
                                Opcode.GT: 'ogt',
                                Opcode.LTE: 'ole',
                                Opcode.GTE: 'oge'
                            }
                            lines.append(f"  %{self.get_temp_reg()} = fcmp {cond_map[opcode]} double {val1}, {val2}")
                        else:
                            cond_map = {
                                Opcode.EQ: 'eq',
                                Opcode.NEQ: 'ne',
                                Opcode.LT: 'slt',
                                Opcode.GT: 'sgt',
                                Opcode.LTE: 'sle',
                                Opcode.GTE: 'sge'
                            }
                            temp_res = self.get_temp_reg()
                            lines.append(f"  %{temp_res} = icmp {cond_map[opcode]} {t1} {val1}, {val2}")
                            llvm_stack.append(f"i1 %{temp_res}")
                            
                elif opcode == Opcode.JMP:
                    lines.append(f"  br label %B{block.successors[0].id}")
                    
                elif opcode == Opcode.JMP_IF_FALSE:
                    if llvm_stack:
                        cond_op = llvm_stack.pop()
                        _, cond_val = cond_op.split(' ', 1)
                        branch_target = block.successors[0].id
                        fall_through = block.successors[1].id
                        lines.append(f"  br i1 {cond_val}, label %B{fall_through}, label %B{branch_target}")
                        
                elif opcode == Opcode.JMP_IF_TRUE:
                    if llvm_stack:
                        cond_op = llvm_stack.pop()
                        _, cond_val = cond_op.split(' ', 1)
                        branch_target = block.successors[0].id
                        fall_through = block.successors[1].id
                        lines.append(f"  br i1 {cond_val}, label %B{branch_target}, label %B{fall_through}")
                        
                elif opcode == Opcode.PRINT:
                    if llvm_stack:
                        val_op = llvm_stack.pop()
                        t, val = val_op.split(' ', 1)
                        if t == 'i32':
                            lines.append(f"  call void @print_int(i32 {val})")
                        elif t == 'double':
                            lines.append(f"  call void @print_double(double {val})")
                        elif t == 'i1':
                            lines.append(f"  call void @print_bool(i1 {val})")
                            
                elif opcode == Opcode.Q_ALLOC:
                    lines.append(f"  %{arg} = call %Qubit* @__quantum__rt__qubit_allocate()")
                    
                elif opcode == Opcode.Q_GATE:
                    gate_name, targets = arg
                    gate_upper = gate_name.upper()
                    params = []
                    if gate_upper in ('RX', 'RY', 'RZ'):
                        if llvm_stack:
                            param_op = llvm_stack.pop()
                            _, param_val = param_op.split(' ', 1)
                            params.append(param_val)
                            
                    q_args = [f"%Qubit* %{t}" for t in targets]
                    if gate_upper == 'H':
                        lines.append(f"  call void @__quantum__qis__h__body({q_args[0]})")
                    elif gate_upper == 'X':
                        lines.append(f"  call void @__quantum__qis__x__body({q_args[0]})")
                    elif gate_upper == 'Y':
                        lines.append(f"  call void @__quantum__qis__y__body({q_args[0]})")
                    elif gate_upper == 'Z':
                        lines.append(f"  call void @__quantum__qis__z__body({q_args[0]})")
                    elif gate_upper == 'S':
                        lines.append(f"  call void @__quantum__qis__s__body({q_args[0]})")
                    elif gate_upper == 'T':
                        lines.append(f"  call void @__quantum__qis__t__body({q_args[0]})")
                    elif gate_upper == 'RX':
                        lines.append(f"  call void @__quantum__qis__rx__body(double {params[0]}, {q_args[0]})")
                    elif gate_upper == 'RY':
                        lines.append(f"  call void @__quantum__qis__ry__body(double {params[0]}, {q_args[0]})")
                    elif gate_upper == 'RZ':
                        lines.append(f"  call void @__quantum__qis__rz__body(double {params[0]}, {q_args[0]})")
                    elif gate_upper == 'CNOT':
                        lines.append(f"  call void @__quantum__qis__cnot__body({q_args[0]}, {q_args[1]})")
                    elif gate_upper == 'CZ':
                        lines.append(f"  call void @__quantum__qis__cz__body({q_args[0]}, {q_args[1]})")
                        
                elif opcode == Opcode.Q_MEASURE:
                    qubit_name, cbit_name = arg
                    lines.append(f"  %res_{cbit_name} = call %Result* @__quantum__qis__m__body(%Qubit* %{qubit_name})")
                    lines.append(f"  %{cbit_name} = call i1 @__quantum__rt__result_get_one(%Result* %res_{cbit_name})")
                    
                elif opcode == Opcode.HALT:
                    lines.append("  ret void")
                    
            lines.append("")
            
        lines.append("}")
        lines.append("")
        
        # Declarations
        lines.append("declare %Qubit* @__quantum__rt__qubit_allocate()")
        lines.append("declare void @__quantum__rt__qubit_release(%Qubit*)")
        lines.append("declare void @__quantum__qis__h__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__x__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__y__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__z__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__s__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__t__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__rx__body(double, %Qubit*)")
        lines.append("declare void @__quantum__qis__ry__body(double, %Qubit*)")
        lines.append("declare void @__quantum__qis__rz__body(double, %Qubit*)")
        lines.append("declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)")
        lines.append("declare void @__quantum__qis__cz__body(%Qubit*, %Qubit*)")
        lines.append("declare %Result* @__quantum__qis__m__body(%Qubit*)")
        lines.append("declare i1 @__quantum__rt__result_get_one(%Result*)")
        lines.append("declare void @print_int(i32)")
        lines.append("declare void @print_double(double)")
        lines.append("declare void @print_bool(i1)")
        
        return "\n".join(lines)
