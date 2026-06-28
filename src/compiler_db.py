import os
import json
import hashlib
import pickle

class QueryDb:
    def __init__(self, workspace_root):
        self.workspace_root = workspace_root
        self.cache_dir = os.path.join(workspace_root, ".eigen_cache")
        self.db_path = os.path.join(self.cache_dir, "db.json")
        self.major = 3
        self.minor = 0
        self.records = {}
        self.current_query = None
        self.query_stack = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("major") == self.major and data.get("minor") == self.minor:
                    self.records = data.get("records", {})
            except Exception:
                self.records = {}

    def save(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({
                    "major": self.major,
                    "minor": self.minor,
                    "records": self.records
                }, f, indent=2)
        except Exception:
            pass

    def get_file_hash(self, filepath):
        if not os.path.exists(filepath):
            return None
        hasher = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except Exception:
            return None

    def execute_query(self, query_name, key, func, *args, **kwargs):
        query_key = f"{query_name}:{key}"
        
        # Track dependency if we are inside another query
        if self.current_query and self.current_query != query_key:
            if query_key not in self.records[self.current_query]["dependencies"]:
                self.records[self.current_query]["dependencies"].append(query_key)

        # Push to stack
        self.query_stack.append(self.current_query)
        self.current_query = query_key
        
        # Initialize record if not present
        if query_key not in self.records:
            self.records[query_key] = {
                "result_hash": "",
                "cache_file": "",
                "dependencies": [],
                "input_files": {}
            }
            
        # Try to use cache
        cached_val = self.get_cached_value(query_key)
        if cached_val is not None:
            # Pop stack
            self.current_query = self.query_stack.pop()
            return cached_val

        # Cache miss: reset dependencies for new execution
        self.records[query_key]["dependencies"] = []

        # Execute
        result = func(*args, **kwargs)

        # Save result to file
        result_hash = self.compute_result_hash(result)
        cache_file = os.path.join(self.cache_dir, f"v3.1_{result_hash}.{query_name}")
        
        # Write cache file
        self.write_cache_file(cache_file, query_name, result)
        
        # Update record
        self.records[query_key]["result_hash"] = result_hash
        try:
            rel_cache = os.path.relpath(cache_file, self.workspace_root)
        except ValueError:
            rel_cache = os.path.abspath(cache_file)
        self.records[query_key]["cache_file"] = rel_cache
        
        # Pop stack
        self.current_query = self.query_stack.pop()
        
        self.save()
        return result

    def get_cached_value(self, query_key):
        record = self.records.get(query_key)
        if not record or not record.get("cache_file") or not record.get("result_hash"):
            return None
            
        # Verify inputs and dependencies
        valid, _ = self.verify_record(query_key)
        if not valid:
            return None
            
        # Load from cache file
        cache_path = os.path.join(self.workspace_root, record["cache_file"])
        if not os.path.exists(cache_path) or os.path.isdir(cache_path):
            return None
            
        return self.read_cache_file(cache_path, query_key.split(":")[0])

    def verify_record(self, query_key):
        record = self.records.get(query_key)
        if not record or not record.get("cache_file") or not record.get("result_hash"):
            return False, "Cache entry missing or incomplete"
            
        # Verify input files
        for filepath, expected_hash in record["input_files"].items():
            # Check either absolute path or relative path
            full_path = filepath if os.path.isabs(filepath) else os.path.join(self.workspace_root, filepath)
            actual_hash = self.get_file_hash(full_path)
            if actual_hash != expected_hash:
                return False, f"Input file '{filepath}' has changed (expected {expected_hash[:8] if expected_hash else None}, got {actual_hash[:8] if actual_hash else None})"
                
        # Verify dependencies
        for dep in record["dependencies"]:
            dep_valid, reason = self.verify_record(dep)
            if not dep_valid:
                return False, f"Dependency '{dep}' is invalid: {reason}"
                
        return True, "Valid"

    def explain_cache(self, query_name, key):
        query_key = f"{query_name}:{key}"
        valid, reason = self.verify_record(query_key)
        if valid:
            print(f"[Explain Cache] Hit: {query_key} is up-to-date.")
        else:
            print(f"[Explain Cache] Miss: {query_key} is invalid. Reason: {reason}")

    def add_input_file(self, filepath):
        if self.current_query:
            h = self.get_file_hash(filepath)
            if h:
                try:
                    rel_path = os.path.relpath(filepath, self.workspace_root)
                except ValueError:
                    rel_path = os.path.abspath(filepath)
                self.records[self.current_query]["input_files"][rel_path] = h

    def compute_result_hash(self, result):
        hasher = hashlib.sha256()
        if isinstance(result, (str, bytes)):
            if isinstance(result, str):
                hasher.update(result.encode('utf-8'))
            else:
                hasher.update(result)
        elif hasattr(result, 'to_dict'):
            hasher.update(json.dumps(result.to_dict(), sort_keys=True).encode('utf-8'))
        elif isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'to_dict'):
            hasher.update(json.dumps([x.to_dict() for x in result], sort_keys=True).encode('utf-8'))
        else:
            try:
                hasher.update(pickle.dumps(result))
            except Exception:
                hasher.update(str(result).encode('utf-8'))
        return hasher.hexdigest()

    def write_cache_file(self, path, query_name, result):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if query_name in ("parse", "resolve_imports", "type_check", "to_eqir"):
            with open(path, "wb") as f:
                pickle.dump(result, f)
        elif query_name in ("to_ssa", "optimize", "to_ebc"):
            if query_name == "to_ebc":
                data = {
                    "major": 3,
                    "minor": 0,
                    "instructions": [inst.to_dict() for inst in result]
                }
            elif query_name == "to_ssa":
                data = {block_id: [inst.to_dict() for inst in block.instructions] for block_id, block in result[0].items()}
            elif query_name == "optimize":
                data = {
                    "major": 3,
                    "minor": 0,
                    "instructions": [inst.to_dict() for inst in result]
                }
            else:
                data = result
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def read_cache_file(self, path, query_name):
        if query_name in ("parse", "resolve_imports", "type_check", "to_eqir"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif query_name in ("to_ssa", "optimize", "to_ebc"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if query_name in ("to_ebc", "optimize"):
                from src.backend.bytecode import Instruction
                if isinstance(data, dict) and "instructions" in data:
                    return [Instruction.from_dict(d) for d in data["instructions"]]
                elif isinstance(data, list):
                    return [Instruction.from_dict(d) for d in data]
            elif query_name == "to_ssa":
                from src.backend.bytecode import Instruction
                from src.ir.ssa.cfg import CFGBlock
                blocks = {}
                for block_id, inst_list in data.items():
                    block = CFGBlock(block_id)
                    block.instructions = [Instruction.from_dict(d) for d in inst_list]
                    blocks[block_id] = block
                return blocks, None
        return None
