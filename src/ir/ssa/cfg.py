# Control Flow Graph (CFG) builder for Eigen SSA IR
from src.backend.bytecode import Opcode, Instruction

class BasicBlock:
    def __init__(self, block_id: int):
        self.id = block_id
        self.instructions = []
        self.predecessors = []
        self.successors = []
        self.phi_nodes = {}  # var_name -> list of (pred_block_id, val)

    def __repr__(self) -> str:
        preds = [p.id for p in self.predecessors]
        succs = [s.id for s in self.successors]
        phi_str = f" Phi: {self.phi_nodes}" if self.phi_nodes else ""
        return f"Block {self.id} (preds={preds}, succs={succs}){phi_str}"

class CFGBuilder:
    def build_cfg(self, instructions: list[Instruction]) -> list[BasicBlock]:
        if not instructions:
            return []

        # 1. Identify leaders (start indices of basic blocks)
        leaders = {0}
        for idx, inst in enumerate(instructions):
            if inst.opcode in (Opcode.JMP, Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE, Opcode.CALL):
                # Target is a leader
                if isinstance(inst.arg, int):
                    leaders.add(inst.arg)
                elif isinstance(inst.arg, tuple) and isinstance(inst.arg[0], int):
                    leaders.add(inst.arg[0])
                # Instruction after jump is a leader
                if idx + 1 < len(instructions):
                    leaders.add(idx + 1)
            elif inst.opcode in (Opcode.RET, Opcode.HALT):
                if idx + 1 < len(instructions):
                    leaders.add(idx + 1)

        sorted_leaders = sorted(list(leaders))
        
        # Create blocks
        blocks = []
        block_by_start = {}
        for i, start in enumerate(sorted_leaders):
            end = sorted_leaders[i+1] if i + 1 < len(sorted_leaders) else len(instructions)
            block = BasicBlock(i)
            block.instructions = instructions[start:end]
            block.start_idx = start
            block.end_idx = end
            blocks.append(block)
            block_by_start[start] = block

        # 2. Build edges (successors / predecessors)
        for block in blocks:
            if not block.instructions:
                continue
            last_inst = block.instructions[-1]
            opcode = last_inst.opcode
            arg = last_inst.arg
            
            if opcode == Opcode.JMP:
                if arg in block_by_start:
                    target = block_by_start[arg]
                    block.successors.append(target)
                    target.predecessors.append(block)
            elif opcode in (Opcode.JMP_IF_FALSE, Opcode.JMP_IF_TRUE):
                # Branch target
                if arg in block_by_start:
                    target = block_by_start[arg]
                    block.successors.append(target)
                    target.predecessors.append(block)
                # Fall through target
                fall_through = block.end_idx
                if fall_through in block_by_start:
                    target = block_by_start[fall_through]
                    block.successors.append(target)
                    target.predecessors.append(block)
            elif opcode in (Opcode.RET, Opcode.HALT):
                # No successors (ends control flow)
                pass
            else:
                # Fall through to next block
                fall_through = block.end_idx
                if fall_through in block_by_start:
                    target = block_by_start[fall_through]
                    block.successors.append(target)
                    target.predecessors.append(block)
                    
        return blocks
