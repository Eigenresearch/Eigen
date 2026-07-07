"""§12.3 part 2 — Seed management for reproducibility.

Roadmap (`sol.md` "12.3 Reproducibility"):

    - [x] Experiment tracking       — covered by `experiment_tracker.py`
    - [x] Result caching            — covered by `experiment_tracker.py`
    - [x] Export to papers           — covered by `experiment_tracker.py`
                                       (LaTeX longtable + JSON dump)
    - [x] Seed management            — covered by this module:
                                          - `SeedManager(master_seed)` — a
                                            stateless per-component seed
                                            deriver using SHA-256.
                                          - `GlobalSeedManager` — process
                                            singleton with overrides.
                                          - Module-level shims:
                                            `set_global_seed`, `get_seed_for`,
                                            `set_component_seed`, `reset_global_seed`.

Components like the simulator, the RNG scheduler, the noise model,
the measurement RNG and the audit-trail sampler all need their own
deterministic bit of entropy. From a single master seed (typically
supplied via `EigenVM(seed=N, deterministic=True)`), this module
derives a unique 64-bit integer for each named component, so the
whole pipeline reproducible from one master without manual seed
plumbing in every constructor.
"""
from __future__ import annotations

import hashlib
import threading
import typing


_INT_BYTES = 8  # 64-bit child seeds


class SeedManager:
    """Derive deterministic per-component seeds from a master seed.

    Each call to ``seed_for(component_name)`` returns a stable 64-bit
    integer obtained via SHA-256 over ``(master_seed, component_name)``;
    the same ``(master, component)`` pair always returns the same
    seed, while distinct pairs collide with negligible probability.

    The manager is stateless — callers can construct a fresh instance
    any time and obtain the same seed stream.
    """

    def __init__(self, master_seed: int):
        if not isinstance(master_seed, int):
            raise TypeError(
                f"master_seed must be int (got "
                f"{type(master_seed).__name__})")
        if isinstance(master_seed, bool):
            raise TypeError("master_seed must be int, not bool")
        if master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        self._master = master_seed

    @property
    def master_seed(self) -> int:
        return self._master

    def seed_for(self, component_name: str) -> int:
        """Return a deterministic 64-bit child seed derived from
        ``master_seed`` and ``component_name``."""
        if not isinstance(component_name, str):
            raise TypeError("component_name must be str")
        h = hashlib.sha256()
        h.update(self._master.to_bytes(8, "big", signed=False))
        h.update(b":")
        h.update(component_name.encode("utf-8"))
        return int.from_bytes(h.digest()[:_INT_BYTES], "big", signed=False)

    def derive_submanager(self, component_name: str) -> "SeedManager":
        """Construct a nested ``SeedManager`` whose master seed is
        the child seed for ``component_name``. Useful for
        hierarchical subcomponents (e.g. a `noise` manager whose
        `qubit` and `readout` sub-channels each get their own seed)."""
        return SeedManager(self.seed_for(component_name))

    def child_seeds(self, *component_names: str) \
            -> typing.Dict[str, int]:
        """Convenience: derive and return a dict of
        ``{component_name: child_seed}`` for the supplied names."""
        return {name: self.seed_for(name) for name in component_names}


class GlobalSeedManager:
    """Process-wide singleton seed manager with optional per-component
    overrides.

    The first call to ``set_global_seed(seed)`` establishes the master;
    subsequent calls reset and replace the master (and clear the
    overrides, since the per-component seeds have changed).

    Each call to ``get_seed_for(component)`` returns the per-component
    seed — either the override if one was registered with
    ``set_component_seed``, or the derivation from the global master.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._master: typing.Optional[int] = None
        self._overrides: typing.Dict[str, int] = {}

    # ------------------------------------------------- master

    def set_global_seed(self, seed: int) -> None:
        with self._lock:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise TypeError(f"seed must be int (got "
                                  f"{type(seed).__name__})")
            if seed < 0:
                raise ValueError("seed must be non-negative")
            self._master = seed
            self._overrides = {}

    def get_global_seed(self) -> typing.Optional[int]:
        with self._lock:
            return self._master

    # ------------------------------------------------- components

    def set_component_seed(self, component_name: str, seed: int) -> None:
        with self._lock:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise TypeError(f"seed must be int (got "
                                  f"{type(seed).__name__})")
            if seed < 0:
                raise ValueError("seed must be non-negative")
            self._overrides[component_name] = seed

    def clear_component_seed(self, component_name: str) -> None:
        with self._lock:
            self._overrides.pop(component_name, None)

    def get_seed_for(self, component_name: str) -> int:
        with self._lock:
            if component_name in self._overrides:
                return self._overrides[component_name]
            if self._master is None:
                raise RuntimeError(
                    "No global seed set; call set_global_seed(seed) "
                    "or set_component_seed(name, seed) to get a "
                    "reproducible seed.")
            return SeedManager(self._master).seed_for(component_name)

    # ------------------------------------------------- reset

    def reset(self) -> None:
        with self._lock:
            self._master = None
            self._overrides = {}

    def is_initialized(self) -> bool:
        with self._lock:
            return self._master is not None or bool(self._overrides)


# Module-level singleton accessor
_GLOBAL_REGISTRY = GlobalSeedManager()


def set_global_seed(seed: int) -> None:
    """Set the process-wide master seed."""
    _GLOBAL_REGISTRY.set_global_seed(seed)


def get_global_seed() -> typing.Optional[int]:
    """Return the process-wide master seed, or ``None`` if not set."""
    return _GLOBAL_REGISTRY.get_global_seed()


def set_component_seed(component_name: str, seed: int) -> None:
    """Override the per-component seed for ``component_name``."""
    _GLOBAL_REGISTRY.set_component_seed(component_name, seed)


def clear_component_seed(component_name: str) -> None:
    """Remove any per-component override for ``component_name``."""
    _GLOBAL_REGISTRY.clear_component_seed(component_name)


def get_seed_for(component_name: str) -> int:
    """Return the deterministic seed for ``component_name`` — either
    the registered override, or the SHA-256 derivation from the
    global master. Raises ``RuntimeError`` if no master and no
    override are set."""
    return _GLOBAL_REGISTRY.get_seed_for(component_name)


def reset_global_seed() -> None:
    """Clear the global seed manager entirely (master + overrides)."""
    _GLOBAL_REGISTRY.reset()


def get_global_registry() -> GlobalSeedManager:
    """Return the module-wide singleton `GlobalSeedManager` instance.
    Tests can reach in and inspect state directly if they prefer."""
    return _GLOBAL_REGISTRY


__all__ = [
    "SeedManager",
    "GlobalSeedManager",
    "set_global_seed",
    "get_global_seed",
    "set_component_seed",
    "clear_component_seed",
    "get_seed_for",
    "reset_global_seed",
    "get_global_registry",
]
