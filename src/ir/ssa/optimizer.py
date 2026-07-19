# SSA Optimization Passes for Eigen 2.3 — Helios
from src.backend.bytecode import Opcode, Instruction
from src.ir.ssa.ssa_builder import SSABuilder
from src.ir.ssa.cfg import BasicBlock

try:
    import eigen_native as native
except ImportError:
    native = None

class SSAOptimizer:
    def __init__(self):
        self.constants = {}  # var_name -> constant_value
        self.copies = {}     # var_name -> original_var_name
        self.expressions = {} # (var1, var2, opcode) -> temp_var_name
        self.used_vars = set()

    def optimize(self, blocks: list[BasicBlock]) -> list[BasicBlock]:
        """Perform multiple optimization passes on SSA BasicBlocks."""
        # 1. Reset state
        self.constants = {}
        self.copies = {}
        self.expressions = {}
        self.used_vars = set()

        # Run passes
        self._analyze_phi_constants(blocks)
        self._propagate_and_fold(blocks)
        self._common_subexpression_elimination(blocks)
        self._copy_propagation(blocks)
        self._dead_store_elimination(blocks)
        self._sparse_conditional_constant_propagation(blocks)

        return blocks

    def _analyze_phi_constants(self, blocks: list[BasicBlock]):
        """If all operands of a phi node are the same constant, propagate it."""
        changed = True
        while changed:
            changed = False
            for block in blocks:
                for phi_var, operands in list(block.phi_nodes.items()):
                    if phi_var in self.constants:
                        continue
                    if not operands:
                        continue
                    
                    # Check if all incoming values are the same constant or map to the same constant
                    first_val = operands[0][1]
                    resolved_const = self.constants.get(first_val, first_val)
                    
                    all_same = True
                    for _pred_id, val in operands:
                        resolved = self.constants.get(val, val)
                        if resolved != resolved_const:
                            all_same = False
                            break
                    
                    # If the resolved value is a constant literal (not a variable name string ending in _version)
                    if all_same and not (isinstance(resolved_const, str) and "_" in resolved_const):
                        self.constants[phi_var] = resolved_const
                        # Delete the phi node as it is resolved
                        del block.phi_nodes[phi_var]
                        changed = True

    def _propagate_and_fold(self, blocks: list[BasicBlock]):
        """Perform constant propagation and local constant folding."""
        for block in blocks:
            new_instrs = []
            idx = 0
            instrs = block.instructions
            
            while idx < len(instrs):
                inst = instrs[idx]
                
                # Constant Propagation
                if inst.opcode == Opcode.LOAD_VAR:
                    var_name = inst.arg
                    if var_name in self.constants:
                        new_instrs.append(Instruction(Opcode.LOAD_CONST, self.constants[var_name], inst.line))
                        idx += 1
                        continue
                    elif var_name in self.copies:
                        inst.arg = self.copies[var_name]

                # Record constants from: LOAD_CONST val, STORE_VAR var
                if (inst.opcode == Opcode.STORE_VAR and 
                    new_instrs and 
                    new_instrs[-1].opcode == Opcode.LOAD_CONST):
                    var_name = inst.arg
                    const_val = new_instrs[-1].arg
                    self.constants[var_name] = const_val

                # Record copies from: LOAD_VAR var1, STORE_VAR var2
                if (inst.opcode == Opcode.STORE_VAR and 
                    new_instrs and 
                    new_instrs[-1].opcode == Opcode.LOAD_VAR):
                    var2 = inst.arg
                    var1 = new_instrs[-1].arg
                    self.copies[var2] = var1

                new_instrs.append(inst)
                
                # Constant Folding: LOAD_CONST v1, LOAD_CONST v2, OP
                if len(new_instrs) >= 3:
                    i1, i2, i3 = new_instrs[-3], new_instrs[-2], new_instrs[-1]
                    if i1.opcode == Opcode.LOAD_CONST and i2.opcode == Opcode.LOAD_CONST:
                        folded_val = None
                        v1, v2 = i1.arg, i2.arg
                        op = i3.opcode
                        
                        try:
                            if op == Opcode.ADD: folded_val = v1 + v2
                            elif op == Opcode.SUB: folded_val = v1 - v2
                            elif op == Opcode.MUL: folded_val = v1 * v2
                            elif op == Opcode.DIV:
                                if v2 == 0:
                                    folded_val = None
                                elif isinstance(v1, int) and isinstance(v2, int):
                                    if v1 == -9223372036854775808 and v2 == -1:
                                        folded_val = None
                                    else:
                                        folded_val = v1 // v2
                                else:
                                    folded_val = v1 / v2
                            elif op == Opcode.EQ: folded_val = v1 == v2
                            elif op == Opcode.NEQ: folded_val = v1 != v2
                            elif op == Opcode.LT: folded_val = v1 < v2
                            elif op == Opcode.GT: folded_val = v1 > v2
                            elif op == Opcode.LTE: folded_val = v1 <= v2
                            elif op == Opcode.GTE: folded_val = v1 >= v2
                            elif op == Opcode.AND: folded_val = bool(v1 and v2)
                            elif op == Opcode.OR: folded_val = bool(v1 or v2)
                        except Exception:
                            pass
                            
                        if folded_val is not None:
                            # Replace the three instructions with single LOAD_CONST
                            new_instrs.pop()
                            new_instrs.pop()
                            new_instrs.pop()
                            new_instrs.append(Instruction(Opcode.LOAD_CONST, folded_val, i3.line))
                
                idx += 1
            block.instructions = new_instrs

    def _common_subexpression_elimination(self, blocks: list[BasicBlock]):
        """Eliminate duplicate computations by replacing them with stored temp variables."""
        for block in blocks:
            new_instrs = []
            idx = 0
            instrs = block.instructions
            
            while idx < len(instrs):
                inst = instrs[idx]
                
                # Check for LOAD_VAR v1, LOAD_VAR v2, OP, STORE_VAR t
                if (idx + 3 < len(instrs) and
                    inst.opcode == Opcode.LOAD_VAR and
                    instrs[idx+1].opcode == Opcode.LOAD_VAR and
                    instrs[idx+2].opcode in (
                        Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV,
                        Opcode.EQ, Opcode.NEQ, Opcode.LT, Opcode.GT
                    ) and
                    instrs[idx+3].opcode == Opcode.STORE_VAR):
                    
                    v1 = inst.arg
                    v2 = instrs[idx+1].arg
                    op = instrs[idx+2].opcode
                    target_var = instrs[idx+3].arg
                    
                    expr_key = (v1, v2, op)
                    if expr_key in self.expressions:
                        # Replace the computation with LOAD_VAR of already computed temp variable
                        temp_var = self.expressions[expr_key]
                        new_instrs.append(Instruction(Opcode.LOAD_VAR, temp_var, inst.line))
                        new_instrs.append(Instruction(Opcode.STORE_VAR, target_var, instrs[idx+3].line))
                        idx += 4
                        continue
                    else:
                        self.expressions[expr_key] = target_var
                
                new_instrs.append(inst)
                idx += 1
            block.instructions = new_instrs

    def _copy_propagation(self, blocks: list[BasicBlock]):
        """Substitute copy variables with their sources."""
        for block in blocks:
            for inst in block.instructions:
                if inst.opcode == Opcode.LOAD_VAR:
                    var_name = inst.arg
                    # Follow copy chain
                    while var_name in self.copies:
                        var_name = self.copies[var_name]
                    inst.arg = var_name

    def _dead_store_elimination(self, blocks: list[BasicBlock]):
        """Remove stores to variables that are never read/used downstream."""
        # 1. Collect all used variables and defined variables in instructions and phi nodes
        self.used_vars = set()
        defined_vars = []
        for block in blocks:
            for inst in block.instructions:
                if inst.opcode == Opcode.LOAD_VAR:
                    self.used_vars.add(inst.arg)
                elif inst.opcode == Opcode.STORE_VAR:
                    defined_vars.append(inst.arg)
            for phi_var, operands in block.phi_nodes.items():
                self.used_vars.add(phi_var)
                for _pred_id, val in operands:
                    self.used_vars.add(val)

        if native is not None:
            unused_set = set(native.fast_unused_vars(defined_vars, list(self.used_vars)))
        else:
            unused_set = set(var for var in defined_vars if var not in self.used_vars)

        # 2. Eliminate dead stores
        for block in blocks:
            new_instrs = []
            idx = 0
            instrs = block.instructions
            while idx < len(instrs):
                inst = instrs[idx]
                if inst.opcode == Opcode.STORE_VAR:
                    var_name = inst.arg
                    # If this store is to an unused variable and preceded by a pure load
                    if (
                        var_name in unused_set
                        and new_instrs
                        and new_instrs[-1].opcode in (Opcode.LOAD_CONST, Opcode.LOAD_VAR)
                    ):
                        new_instrs.pop()  # Remove the producer instruction
                        idx += 1
                        continue
                new_instrs.append(inst)
                idx += 1
            block.instructions = new_instrs

    def _sparse_conditional_constant_propagation(self, blocks: list[BasicBlock]):
        """Evaluate constant branch conditions and convert to unconditional jumps."""
        for block in blocks:
            if len(block.instructions) >= 2:
                i1, i2 = block.instructions[-2], block.instructions[-1]
                if i1.opcode == Opcode.LOAD_CONST and i2.opcode in (Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE):
                    cond_val = bool(i1.arg)
                    jump_target = i2.arg
                    
                    # Determine if branch is taken
                    taken = False
                    if i2.opcode == Opcode.JMP_IF_FALSE and not cond_val:
                        taken = True
                    elif i2.opcode == Opcode.JMP_IF_TRUE and cond_val:
                        taken = True
                        
                    # Remove condition load and branch instruction
                    block.instructions.pop()
                    block.instructions.pop()
                    
                    if taken:
                        # Replace with unconditional jump
                        block.instructions.append(Instruction(Opcode.JMP, jump_target, i2.line))
                    else:
                        # Fallthrough (do nothing, it will execute next block in order)
                        pass


def optimize_ebc(instructions: list[Instruction], de_ssa: bool = True) -> list[Instruction]:
    """Convert EBC to SSA form, run optimizer passes, and rebuild EBC."""
    builder = SSABuilder()
    # 1. Build SSA CFG blocks
    blocks, _ = builder.build_ssa(instructions)
    if not blocks:
        return instructions

    # Save original block properties for target remapping
    original_blocks = [BasicBlock(b.id) for b in blocks]
    for b_orig, b_curr in zip(original_blocks, blocks, strict=False):
        b_orig.start_idx = b_curr.start_idx
        b_orig.end_idx = b_curr.end_idx

    # 2. Optimize blocks
    optimizer = SSAOptimizer()
    optimized_blocks = optimizer.optimize(blocks)

    # 3. Rebuild instructions from blocks
    optimized_instructions = []
    block_new_starts = {}
    
    for block in optimized_blocks:
        block_new_starts[block.id] = len(optimized_instructions)
        optimized_instructions.extend(block.instructions)
        
    # Map old start index to block ID
    old_start_to_id = {b.start_idx: b.id for b in original_blocks}

    # 4. Remap jump targets to new instruction offsets
    for inst in optimized_instructions:
        if inst.opcode in (Opcode.JMP, Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE):
            old_target = inst.arg
            if isinstance(old_target, int):
                block_id = old_start_to_id.get(old_target)
                if block_id is not None and block_id in block_new_starts:
                    inst.arg = block_new_starts[block_id]
            elif isinstance(old_target, tuple) and isinstance(old_target[0], int):
                old_ip, name, num_args = old_target
                block_id = old_start_to_id.get(old_ip)
                if block_id is not None and block_id in block_new_starts:
                    inst.arg = (block_new_starts[block_id], name, num_args)

    if de_ssa:
        # 5. Strip SSA suffixes from variable names
        import re
        def strip_var(var):
            if not isinstance(var, str):
                return var
            m = re.search(r'_(\d+)$', var)
            if m:
                return var[:-len(m.group(0))]
            return var

        for inst in optimized_instructions:
            if inst.opcode == Opcode.LOAD_VAR:
                inst.arg = strip_var(inst.arg)
            elif inst.opcode == Opcode.STORE_VAR:
                inst.arg = strip_var(inst.arg)
            elif inst.opcode == Opcode.Q_MEASURE:
                if isinstance(inst.arg, tuple) and len(inst.arg) == 2:
                    q, cbit = inst.arg
                    inst.arg = (q, strip_var(cbit))

    return optimized_instructions
