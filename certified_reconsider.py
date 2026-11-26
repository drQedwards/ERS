"""Optional exact-certificate gate for ERS reconsideration.

This module does not replace ``reconsider_memory`` in ERS.py. It is a
stdlib-only companion: if a caller has a finite certificate, they can
refuse to promote or rewrite a memory when verification fails.

Supported certificate kinds (all finite, ``clay_proof = 0``):

- ``gf2_farkas_unsat``: a list of GF(2) rows ``(mask, rhs)`` and a list
  of row indices that should form a Farkas dual (XOR of selected rows
  has mask 0 and rhs 1).
- ``drup_empty``: a tiny CNF plus a DRUP-style addition trace that must
  end in the empty clause after reverse unit propagation of additions
  only (no deletion steps in this minimal hook).

This is not a proof of an open problem and does not claim that upstream
ERS already emits machine-checked proofs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

FINITE_EXPERIMENT = True
CLAY_PROOF = 0


def _xor_rows(rows: Sequence[Tuple[int, int]], indices: Sequence[int]) -> Tuple[int, int]:
    mask = 0
    rhs = 0
    n = len(rows)
    for i in indices:
        if i < 0 or i >= n:
            raise ValueError("certificate index out of range")
        m, b = rows[i]
        mask ^= int(m)
        rhs ^= int(b) & 1
    return mask, rhs


def verify_gf2_farkas_unsat(rows: Sequence[Tuple[int, int]], indices: Sequence[int]) -> bool:
    if not indices:
        return False
    try:
        mask, rhs = _xor_rows(rows, indices)
    except ValueError:
        return False
    return mask == 0 and rhs == 1


def verify_drup_empty(cnf: Sequence[Sequence[int]], steps: Sequence[Tuple[str, Sequence[int]]]) -> bool:
    """Accept a trace of ('a', lits) that unit-propagates to the empty clause."""
    if not steps:
        return False
    clauses: List[List[int]] = [list(c) for c in cnf]
    assigned: Dict[int, bool] = {}

    def false_lit(lit: int) -> bool:
        v = abs(lit)
        if v not in assigned:
            return False
        return assigned[v] != (lit > 0)

    def unit_propagate() -> str:
        changed = True
        while changed:
            changed = False
            for cl in clauses:
                live = [lit for lit in cl if not false_lit(lit)]
                if not live:
                    return "empty"
                sat = False
                for lit in cl:
                    v = abs(lit)
                    if v in assigned and assigned[v] == (lit > 0):
                        sat = True
                        break
                if sat:
                    continue
                if len(live) == 1:
                    lit = live[0]
                    v = abs(lit)
                    val = lit > 0
                    if v in assigned and assigned[v] != val:
                        return "empty"
                    if v not in assigned:
                        assigned[v] = val
                        changed = True
        return "open"

    if unit_propagate() == "empty":
        return any(op == "a" and len(list(lits)) == 0 for op, lits in steps) or False

    for op, lits in steps:
        if op != "a":
            return False
        clauses.append(list(lits))
        if unit_propagate() == "empty":
            return True
    return False


def verify_reconsider_certificate(certificate: Mapping[str, Any]) -> bool:
    kind = certificate.get("kind")
    if kind == "gf2_farkas_unsat":
        rows = certificate.get("rows") or []
        indices = certificate.get("indices") or []
        packed = [(int(m), int(b)) for m, b in rows]
        return verify_gf2_farkas_unsat(packed, list(indices))
    if kind == "drup_empty":
        cnf = certificate.get("cnf") or []
        steps = certificate.get("steps") or []
        parsed = [(str(op), list(lits)) for op, lits in steps]
        return verify_drup_empty(cnf, parsed)
    return False


def reconsider_with_certificate(certificate: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a small admission record. False verification means do not promote."""
    ok = verify_reconsider_certificate(certificate)
    return {
        "FINITE_EXPERIMENT": FINITE_EXPERIMENT,
        "clay_proof": CLAY_PROOF,
        "verified": ok,
        "admit": ok,
        "kind": certificate.get("kind"),
    }
