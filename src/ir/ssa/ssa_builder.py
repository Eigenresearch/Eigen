# SSA IR Builder: Renaming variables and converting EBC bytecode to SSA form
from src.backend.bytecode import Opcode, Instruction
from src.ir.ssa.cfg import CFGBuilder, BasicBlock
from src.ir.ssa.dominance import DominanceAnalyzer
from src.ir.ssa.phi_nodes import PhiNodePlacer

class SSABuilder:
    def __init__(self):
        self.version_stack = {}  # var_name -> list of int
        self.counters = {}       # var_name -> int
        self.dom_tree = {}       # block_id -> list of block_id (children)

    def get_latest_version(self, var: str) -> str:
        if not isinstance(var, str):
            return var
        import re
        if re.search(r'[\s\(\)\=\!\+\-\*\/\<\>]', var):
            def replace_word(match):
                word = match.group(0)
                if word in ('True', 'False', 'None', 'int', 'float', 'str', 'bool') or word.isdigit():
                    return word
                return self.get_latest_version(word)
            return re.sub(r'[a-zA-Z_][a-zA-Z0-9_]*', replace_word, var)

        if var in self.version_stack and self.version_stack[var]:
            return f"{var}_{self.version_stack[var][-1]}"
        # If not defined yet, default to version 0
        return f"{var}_0"

    def new_version(self, var: str) -> str:
        self.counters[var] = self.counters.get(var, 0) + 1
        ver = self.counters[var]
        if var not in self.version_stack:
            self.version_stack[var] = []
        self.version_stack[var].append(ver)
        return f"{var}_{ver}"

    def build_ssa(self, instructions: list[Instruction]) -> tuple[list[BasicBlock], str]:
        # 1. Build CFG
        cfg_builder = CFGBuilder()
        blocks = cfg_builder.build_cfg(instructions)
        if not blocks:
            return [], ""

        # 2. Dominance Analysis
        dom_analyzer = DominanceAnalyzer(blocks)

        # Build dominator tree children map
        self.dom_tree = {b.id: [] for b in blocks}
        for b in blocks:
            parent = dom_analyzer.idom.get(b.id)
            if parent is not None:
                self.dom_tree[parent].append(b.id)

        # 3. Place Phi-Nodes
        placer = PhiNodePlacer()
        placer.place_phi_nodes(blocks, dom_analyzer)

        # 4. Rename variables recursively
        self.version_stack = {}
        self.counters = {}
        
        self.rename_block(blocks[0], blocks)
        
        # 5. Build string representation
        lines = []
        for b in blocks:
            lines.append(f"Block {b.id}:")
            for var, phi in sorted(b.phi_nodes.items()):
                phi_args = ", ".join(f"B{pred_id}:{val}" for pred_id, val in phi)
                # Output version of the phi node
                # Since the phi is renamed, we can print it
                lines.append(f"  {var} = phi({phi_args})")
            for inst in b.instructions:
                lines.append(f"  {inst}")
        
        return blocks, "\n".join(lines)

    def rename_block(self, block: BasicBlock, blocks: list[BasicBlock]):
        # Track versions pushed in this block to pop later
        pushed = []

        # 1. Rename output variables of phi nodes
        # phi_nodes[var_name] = list of (pred_id, val)
        # We rename the keys to their new versions
        new_phi_nodes = {}
        for var in list(block.phi_nodes.keys()):
            new_var = self.new_version(var)
            pushed.append(var)
            new_phi_nodes[new_var] = block.phi_nodes[var]
        block.phi_nodes = new_phi_nodes

        # 2. Rename variables in instructions
        for inst in block.instructions:
            # If instruction uses variable: LOAD_VAR
            if inst.opcode == Opcode.LOAD_VAR:
                inst.arg = self.get_latest_version(inst.arg)
            # If instruction defines variable: STORE_VAR
            elif inst.opcode == Opcode.STORE_VAR:
                new_var = self.new_version(inst.arg)
                pushed.append(inst.arg)
                inst.arg = new_var
            # If instruction is Q_MEASURE: argument is (qubit, cbit)
            elif inst.opcode == Opcode.Q_MEASURE:
                q, cbit = inst.arg
                new_cbit = self.new_version(cbit)
                pushed.append(cbit)
                inst.arg = (q, new_cbit)

        # 3. Populate phi node arguments in successor blocks
        for succ in block.successors:
            # Succ block has phi nodes keyed by new_var (renamed) or old var?
            # Wait! Since successor's phi nodes are not renamed yet, we must
            # check the original variable names!
            # Since successor's phi nodes get renamed when we visit the successor block,
            # but we are at the predecessor block now. How do we match?
            # Let's map renamed keys in succ back to original name or store original name.
            # Actually, to make this easy, we can keep original variable names in successor's phis until renaming.
            # So in succ.phi_nodes (which is not renamed yet), keys are original names.
            # Let's check succ.phi_nodes:
            for orig_var in list(succ.phi_nodes.keys()):
                # We find the latest version in the predecessor block
                latest = self.get_latest_version(orig_var)
                # Append (predecessor_id, latest_version_name)
                succ.phi_nodes[orig_var].append((block.id, latest))

        # 4. Recursively rename in dominator tree children
        for child_id in self.dom_tree.get(block.id, []):
            child_block = self.get_block_by_id(blocks, child_id)
            self.rename_block(child_block, blocks)

        # 5. Pop versions pushed in this block
        for var in pushed:
            if var in self.version_stack and self.version_stack[var]:
                self.version_stack[var].pop()

    def get_block_by_id(self, blocks: list[BasicBlock], block_id: int) -> BasicBlock:
        for b in blocks:
            if b.id == block_id:
                return b
        return None
