"""§1.2 — Compiler Speed: regex lexer helper, type checker cache,
import cache, incremental AST/EQIR/EBC cache, lazy module loading.

Surface-level optimization infrastructure for the Eigen compiler
pipeline. These are cache and lazy-loading helpers that reduce
recompilation overhead for unchanged modules.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import typing


@dataclasses.dataclass
class CacheEntry:
    """Single entry in the incremental compilation cache."""
    key: str
    artifact_hash: str
    source_path: str
    timestamp: float
    payload: typing.Any = None


class TypeCheckerCache:
    """Caches resolved types by (name, scope_id) to avoid
    re-resolving the same type expression on every reference.

    §1.2: "Кэширование разрешённых типов в type checker"
    """

    __slots__ = ("_cache", "_hits", "_misses")

    def __init__(self):
        self._cache: dict[tuple[str, int], typing.Any] = {}
        self._hits = 0
        self._misses = 0

    def get(self, name: str, scope_id: int) -> typing.Any | None:
        key = (name, scope_id)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            return result
        self._misses += 1
        return None

    def put(self, name: str, scope_id: int, resolved_type: typing.Any):
        self._cache[(name, scope_id)] = resolved_type

    def invalidate_scope(self, scope_id: int):
        keys_to_remove = [k for k in self._cache if k[1] == scope_id]
        for k in keys_to_remove:
            del self._cache[k]

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        return {"entries": len(self._cache), "hits": self._hits,
                  "misses": self._misses}


class ImportCache:
    """Caches imported modules to avoid re-parsing/re-resolving
    unchanged modules.

    §1.2: "Кэш импортов для неизменённых модулей"
    """

    __slots__ = ("_cache", "_file_hashes")

    def __init__(self):
        self._cache: dict[str, typing.Any] = {}
        self._file_hashes: dict[str, str] = {}

    def _file_hash(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except (IOError, OSError):
            return ""

    def get(self, module_path: str, source_file: str | None = None
            ) -> tuple[typing.Any, bool]:
        """Returns (cached_module, is_fresh).
        is_fresh=True means cache is valid; False means stale.
        """
        if source_file and os.path.exists(source_file):
            current_hash = self._file_hash(source_file)
            cached_hash = self._file_hashes.get(source_file)
            if cached_hash == current_hash:
                return self._cache.get(module_path), True
            else:
                self._file_hashes[source_file] = current_hash
                return None, False
        return self._cache.get(module_path), True

    def put(self, module_path: str, module: typing.Any,
            source_file: str | None = None):
        self._cache[module_path] = module
        if source_file and os.path.exists(source_file):
            self._file_hashes[source_file] = self._file_hash(source_file)

    def invalidate(self, module_path: str | None = None):
        if module_path:
            self._cache.pop(module_path, None)
        else:
            self._cache.clear()
            self._file_hashes.clear()

    def stats(self) -> dict:
        return {"cached_modules": len(self._cache),
                  "tracked_files": len(self._file_hashes)}


class IncrementalCache:
    """Incremental cache for AST/EQIR/EBC artifacts.

    §1.2: "Инкрементальный кэш AST/EQIR/EBC"
    Stores compiled artifacts keyed by source hash. On recompile,
    if the source hash matches, the cached artifact is reused.
    """

    __slots__ = ("_ast_cache", "_eqir_cache", "_ebc_cache",
                  "_source_hashes")

    def __init__(self):
        self._ast_cache: dict[str, typing.Any] = {}
        self._eqir_cache: dict[str, typing.Any] = {}
        self._ebc_cache: dict[str, typing.Any] = {}
        self._source_hashes: dict[str, str] = {}

    def _source_hash(self, source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    def get_ast(self, source: str) -> tuple[typing.Any, bool]:
        h = self._source_hash(source)
        key = h
        if key in self._ast_cache and self._source_hashes.get(key) == h:
            return self._ast_cache[key], True
        return None, False

    def put_ast(self, source: str, ast: typing.Any):
        h = self._source_hash(source)
        self._ast_cache[h] = ast
        self._source_hashes[h] = h

    def get_eqir(self, source_hash: str) -> tuple[typing.Any, bool]:
        if source_hash in self._eqir_cache:
            return self._eqir_cache[source_hash], True
        return None, False

    def put_eqir(self, source_hash: str, graph: typing.Any):
        self._eqir_cache[source_hash] = graph

    def get_ebc(self, source_hash: str) -> tuple[typing.Any, bool]:
        if source_hash in self._ebc_cache:
            return self._ebc_cache[source_hash], True
        return None, False

    def put_ebc(self, source_hash: str, bytecode: typing.Any):
        self._ebc_cache[source_hash] = bytecode

    def invalidate(self, source: str | None = None):
        if source:
            h = self._source_hash(source)
            self._ast_cache.pop(h, None)
            self._eqir_cache.pop(h, None)
            self._ebc_cache.pop(h, None)
            self._source_hashes.pop(h, None)
        else:
            self._ast_cache.clear()
            self._eqir_cache.clear()
            self._ebc_cache.clear()
            self._source_hashes.clear()

    def stats(self) -> dict:
        return {"ast_entries": len(self._ast_cache),
                  "eqir_entries": len(self._eqir_cache),
                  "ebc_entries": len(self._ebc_cache)}


class LazyModuleLoader:
    """Lazy module loading — loads modules only when first referenced.

    §1.2: "Ленивая загрузка модулей"
    """

    __slots__ = ("_loaders", "_loaded", "_loading")

    def __init__(self):
        self._loaders: dict[str, typing.Callable] = {}
        self._loaded: dict[str, typing.Any] = {}
        self._loading: set[str] = set()

    def register(self, name: str, loader: typing.Callable):
        self._loaders[name] = loader

    def load(self, name: str) -> typing.Any:
        if name in self._loaded:
            return self._loaded[name]
        if name in self._loading:
            raise RuntimeError(f"Circular lazy load detected for '{name}'")
        if name not in self._loaders:
            raise KeyError(f"No loader registered for module '{name}'")
        self._loading.add(name)
        try:
            module = self._loaders[name]()
            self._loaded[name] = module
            return module
        finally:
            self._loading.discard(name)

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def unload(self, name: str):
        self._loaded.pop(name, None)

    def stats(self) -> dict:
        return {"registered": len(self._loaders),
                  "loaded": len(self._loaded)}


def regex_lexer_tokenize(source: str) -> list[tuple]:
    """Regex-based tokenizer helper for the lexer.

    §1.2: "Переписать лексер на regex/slicing подход (5-10x ускорение)"
    This is a supplementary regex-based tokenizer that can be used
    by the lexer for faster tokenization. Returns list of
    (token_type_str, value, line, col) tuples.
    """
    import re

    # Token patterns in priority order
    patterns = [
        ("WHITESPACE", r"[ \t]+"),
        ("NEWLINE", r"\n"),
        ("COMMENT", r"#[^\n]*"),
        ("FLOAT_LIT", r"\d+\.\d+([eE][+-]?\d+)?"),
        ("INT_LIT", r"\d+"),
        ("STRING_LIT", r'"[^"]*"'),
        ("STRING_LIT_SINGLE", r"'[^']*'"),
        ("ARROW", r"->"),
        ("POW", r"\*\*"),
        ("ADD_ASSIGN", r"\+="),
        ("SUB_ASSIGN", r"-="),
        ("MUL_ASSIGN", r"\*="),
        ("DIV_ASSIGN", r"/="),
        ("LSHIFT", r"<<"),
        ("RSHIFT", r">>"),
        ("LE", r"<="),
        ("GE", r">="),
        ("EQ", r"=="),
        ("NE", r"!="),
        ("AND", r"\band\b"),
        ("OR", r"\bor\b"),
        ("NOT", r"\bnot\b"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MUL", r"\*"),
        ("DIV", r"/"),
        ("MOD", r"%"),
        ("LT", r"<"),
        ("GT", r">"),
        ("ASSIGN", r"="),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LBRACK", r"\["),
        ("RBRACK", r"\]"),
        ("COMMA", r","),
        ("COLON", r":"),
        ("DOT", r"\."),
        ("SEMICOLON", r";"),
        ("AMP", r"&"),
        ("PIPE", r"\|"),
        ("CARET", r"\^"),
        ("TILDE", r"~"),
    ]

    combined = "|".join(f"(?P<{name}>{pat})" for name, pat in patterns)
    regex = re.compile(combined)

    tokens = []
    line = 1
    col = 1
    for m in regex.finditer(source):
        tok_type = m.lastgroup
        value = m.group()
        if tok_type in ("WHITESPACE",):
            col += len(value)
            continue
        if tok_type == "NEWLINE":
            line += 1
            col = 1
            continue
        if tok_type == "COMMENT":
            col += len(value)
            continue
        if tok_type == "STRING_LIT_SINGLE":
            tok_type = "STRING_LIT"
        tokens.append((tok_type, value, line, col))
        col += len(value)

    tokens.append(("EOF", "", line, col))
    return tokens
