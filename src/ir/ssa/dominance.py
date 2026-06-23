# Dominance tree and frontiers calculation for SSA IR
from src.ir.ssa.cfg import BasicBlock

class DominanceAnalyzer:
    def __init__(self, blocks: list[BasicBlock]):
        self.blocks = blocks
        self.dom = {}  # block_id -> set of block_ids
        self.idom = {}  # block_id -> block_id
        self.df = {}  # block_id -> set of block_ids
        self.analyze()

    def analyze(self):
        if not self.blocks:
            return

        # 1. Compute dominators
        entry = self.blocks[0].id
        all_ids = {b.id for b in self.blocks}
        
        self.dom[entry] = {entry}
        for b in self.blocks[1:]:
            self.dom[b.id] = set(all_ids)

        changed = True
        while changed:
            changed = False
            for b in self.blocks[1:]:
                pred_doms = []
                for pred in b.predecessors:
                    if pred.id in self.dom:
                        pred_doms.append(self.dom[pred.id])
                
                if pred_doms:
                    new_dom = {b.id}.union(set.intersection(*pred_doms))
                else:
                    new_dom = {b.id}
                    
                if new_dom != self.dom[b.id]:
                    self.dom[b.id] = new_dom
                    changed = True

        # 2. Compute immediate dominators (idom)
        for b in self.blocks:
            self.df[b.id] = set()
            if b.id == entry:
                self.idom[b.id] = None
                continue
                
            # idom[b.id] is the dominator d != b.id that is dominated by all other dominators
            candidates = self.dom[b.id] - {b.id}
            # Find the candidate that dominates all other candidates
            for c in candidates:
                # If c is dominated by all other candidates (i.e. other candidates are subsets of c's doms)
                # actually, idom is the closest strictly dominating node.
                # It is the unique node in candidates that dominates no other nodes in candidates.
                # So we can count the size of candidates. The one with size len(candidates) - 1 is the idom.
                if len(self.dom[c]) == len(candidates):
                    self.idom[b.id] = c
                    break

        # 3. Compute dominance frontiers (df)
        for b in self.blocks:
            if len(b.predecessors) >= 2:
                for pred in b.predecessors:
                    runner = pred.id
                    while runner != self.idom.get(b.id):
                        if runner in self.df:
                            self.df[runner].add(b.id)
                        else:
                            self.df[runner] = {b.id}
                        runner = self.idom.get(runner)
