#!/usr/bin/env python3
"""Cold-start tests for certified_reconsider.py. Zero bare asserts; works under python -O."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certified_reconsider import reconsider_with_certificate, verify_reconsider_certificate


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def main() -> int:
    unsat = {
        "kind": "gf2_farkas_unsat",
        "rows": [(0b1, 0), (0b1, 1)],
        "indices": [0, 1],
    }
    check(verify_reconsider_certificate(unsat) is True, "honest Farkas unsat rejected")
    check(reconsider_with_certificate(unsat)["admit"] is True, "honest cert not admitted")

    check(verify_reconsider_certificate({**unsat, "indices": []}) is False, "empty Farkas accepted")
    check(verify_reconsider_certificate({**unsat, "indices": [0]}) is False, "single-row Farkas accepted")
    check(verify_reconsider_certificate({**unsat, "indices": [9]}) is False, "OOB index accepted")

    drup = {
        "kind": "drup_empty",
        "cnf": [[1], [-1]],
        "steps": [("a", [1]), ("a", [-1]), ("a", [])],
    }
    check(verify_reconsider_certificate(drup) is True, "honest DRUP rejected")
    check(verify_reconsider_certificate({**drup, "steps": []}) is False, "empty DRUP accepted")
    check(verify_reconsider_certificate({"kind": "nope"}) is False, "unknown kind accepted")

    rec = reconsider_with_certificate({**unsat, "indices": []})
    check(rec["FINITE_EXPERIMENT"] is True and rec["clay_proof"] == 0, "honesty labels missing")
    check(rec["admit"] is False, "failed cert was admitted")
    print("PASS certified_reconsider: Farkas + DRUP plants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
