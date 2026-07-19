"""
§7.3 — Parser/Type-Checker "Did you mean?" suggestions.

Audit checklist item:

  > "Did you mean?" предложения

When the user typos an identifier, a gate mnemonic, or a known keyword,
the existing error surface reads literally:

    Type Error: Undeclared variable 'mesaure' at VarRefNode(mesaure)
    Parser Error at line 3, col 5: Expected type name (found Token(IDENTIFIER, 'flaot', line=3, col=5))

That's accurate but unfriendly. This module computes a single nearest
candidate by Levenshtein distance over a supplied vocabulary and emits
a ", did you mean 'X'?" suffix. The caller appends the suggestion to
its existing error message so the surface semantics don't change
(we're not auto-correcting anything — the error still raises).

We use the classic two-row Wagner–Fischer Levenshtein with an early
exit if the candidate is within a configurable distance threshold; the
distance threshold defaults to 3, which empirically catches typos
1-2 edits away plus transpositions without false-matching unrelated
identifiers.
"""
from typing import Iterable, Optional


_DEFAULT_MAX_DISTANCE = 3


def levenshtein(a: str, b: str, max_distance: int = _DEFAULT_MAX_DISTANCE
                ) -> int:
    """Two-row Wagner–Fischer Levenshtein with an early-prune cap.

    Returns the actual distance if it's <= ``max_distance``; returns
    ``max_distance + 1`` if it would exceed the cap (caller treats
    that as "no candidate"). The early prune keeps us at O(min(m, n))
    memory and rejects long irrelevant candidates without walking the
    full matrix.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    if not a:
        return min(len(b), max_distance + 1)
    if not b:
        return min(len(a), max_distance + 1)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i] + [0] * len(b)
        row_min = cur_row[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur_row[j] = min(
                prev_row[j] + 1,         # deletion
                cur_row[j - 1] + 1,      # insertion
                prev_row[j - 1] + cost,  # substitution
            )
            if cur_row[j] < row_min:
                row_min = cur_row[j]
        # Early prune: every replacement of column j requires the
        # previous row at j, which is at least row_min. If row_min
        # already exceeds the cap, the final distance will too.
        if row_min > max_distance:
            return max_distance + 1
        prev_row = cur_row
    return prev_row[-1]


def suggest(name: str, vocabulary: Iterable[str],
            max_distance: int = _DEFAULT_MAX_DISTANCE) -> Optional[str]:
    """Return the closest vocabulary entry within ``max_distance``
    or ``None`` if nothing matches.

    The vocabulary is iterated in insertion order; ties go to the
    earlier-yielded candidate. Case-sensitive — callers that want
    case-insensitive matching should normalise the vocabulary and the
    probe string beforehand.

    A *prefix-shortcut* fires when either ``name`` or ``cand`` is a
    strict prefix of the other; in that case the Levenshtein distance
    is exactly ``abs(len(cand) - len(name))`` (pure insertion/deletion
    of the trailing suffix), so we use that length-diff directly
    instead of computing the matrix. The shortcut is purely a
    performance optimisation — it must respect the cap, otherwise
    trivial-but-too-far prefix matches would leak in.
    """
    if not name:
        return None
    best: Optional[str] = None
    best_d = max_distance + 1
    for cand in vocabulary:
        if not cand:
            continue
        if cand == name:
            return cand
        # Prefix-shortcut: any kind of prefix match yields distance =
        # pure length-diff (pure insertion/deletion of the suffix).
        # Respect the cap — a too-far prefix is not a candidate.
        if cand.startswith(name) or name.startswith(cand):
            d = abs(len(cand) - len(name))
        else:
            d = levenshtein(name, cand, max_distance=max_distance)
        if d <= max_distance and d < best_d:
            best_d = d
            best = cand
    return best


def format_suggestion(name: str, vocabulary: Iterable[str],
                      max_distance: int = _DEFAULT_MAX_DISTANCE) -> str:
    """Return ``", did you mean 'X'?"`` if a close candidate exists,
    otherwise empty string. Suitable for append-to-error-message use."""
    cand = suggest(name, vocabulary, max_distance=max_distance)
    if cand is None or cand == name:
        return ""
    return f", did you mean '{cand}'?"
