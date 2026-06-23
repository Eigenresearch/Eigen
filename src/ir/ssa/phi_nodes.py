# Phi node insertion for SSA IR
from src.ir.ssa.cfg import BasicBlock
from src.ir.ssa.dominance import DominanceAnalyzer
from src.backend.bytecode import Opcode

class PhiNodePlacer:
    def place_phi_nodes(self, blocks: list[BasicBlock], dom_analyzer: DominanceAnalyzer):
        if not blocks:
            return

        # 1. Find all variable definitions
        # var_name -> set of block_ids containing definitions
        defs = {}
        for block in blocks:
            for inst in block.instructions:
                if inst.opcode == Opcode.STORE_VAR:
                    var_name = inst.arg
                    if var_name not in defs:
                        defs[var_name] = set()
                    defs[var_name].add(block.id)

        # 2. Place phi-nodes based on dominance frontiers
        for var_name, def_blocks in defs.items():
            worklist = list(def_blocks)
            added_phi = set()  # block_ids where phi node for this var has been placed
            
            while worklist:
                n = worklist.pop(0)
                # Get dominance frontier of block n
                df_n = dom_analyzer.df.get(n, set())
                for y_id in df_n:
                    if y_id not in added_phi:
                        # Add phi node in block y
                        y_block = self.get_block_by_id(blocks, y_id)
                        # We initialize phi with empty list of (pred_id, val)
                        # which will be filled during renaming
                        y_block.phi_nodes[var_name] = []
                        added_phi.add(y_id)
                        
                        # If y_block was not a definition block, add to worklist
                        if y_id not in def_blocks:
                            worklist.append(y_id)

    def get_block_by_id(self, blocks: list[BasicBlock], block_id: int) -> BasicBlock:
        for b in blocks:
            if b.id == block_id:
                return b
        return None
