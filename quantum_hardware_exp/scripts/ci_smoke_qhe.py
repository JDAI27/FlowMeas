#!/usr/bin/env python3
"""
CI smoke test for quantum_hardware_exp on simulator.

Runs a small-budget DSS comparison and ensures the pipeline executes and writes results.
"""

import sys

from quantum_hardware_exp.runner.compare_dss import main as compare_main


def main() -> int:
    # Small budget to keep CI fast; single repeat
    sys.argv = [
        "prog",
        "--budget",
        "50",
        "--repeats",
        "1",
        "--out",
        "quantum_hardware_exp/results/dss_ci_smoke.json",
    ]
    try:
        compare_main()
        return 0
    except Exception as e:
        print(f"Smoke test failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
