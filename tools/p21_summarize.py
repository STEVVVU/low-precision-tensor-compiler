"""Curate existing clean receipts without running kernels or calculating new timings."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / ".verification-runs")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence/P21/metrics-summary.json")
    args = parser.parse_args()
    record = {
        "environment": "RTX 5080 Laptop SM120, driver592.01, CUDA13.0.88, Torch2.11.0+cu128, TileLang0.1.15, Python3.11; coordinated slot2026-10-22T06:00:08Z to06:01:04Z; user applications preserved",
        "selector": "original heuristic retained; candidate block_m64 and K>=1024 fragment override was reverted because public-call improvement was inconclusive",
        "timing_scope": "Public wall time includes dispatch, allocation and synchronization. CUDA events measure stream spans and may include host enqueue gaps. Graph replay removes the public-operation Python path and host allocation but replay enqueue gaps can remain. Batch p95 is not per-operation tail latency.",
        "raw_sha256": {},
        "public_results": [],
        "numerical_policy": {
            "original_gate": "ATOL0.0625, RTOL0.001, plus maximum two FP16 ULP against rounded CPU float64",
            "expanded_receipts": "executed with ATOL/RTOL and ULP diagnostic, not the original two-ULP hard gate",
            "freeze_boundary": "no independently timestamped evidence approving the expanded ULP override before observations",
            "interpretation": "cases failing two ULP are diagnostic timings under a weaker criterion; no equal-quality speed claim under the original gate",
        },
        "numerical_review": [],
    }
    for label in ("sweep", "held-out"):
        path = args.root / f"P21-metrics-clean-{label}" / "sweep-result.json"
        raw = json.loads(path.read_text())
        if not raw["pass"]:
            raise ValueError("failed receipt")
        record["raw_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for case in raw["results"]:
            if not case["pass"] or case["samples_per_side"] != 7 or case["calls_per_sample"] != 200 or case["counts"]["failed"]:
                raise ValueError("wrong campaign or incomplete work")
            if any(len(timing["samples"]) != 7 for timing in case["wall_clock_latency_ms"].values()):
                raise ValueError("missing samples")
            errors = case["numerical_error_vs_independent_oracles"]
            sides = {
                name: {"max_ulp": error["max_ulp"], "passes_original_two_ulp_gate": error["max_ulp"] <= 2} for name, error in errors.items()
            }
            passes_original = all(side["passes_original_two_ulp_gate"] for side in sides.values())
            record["numerical_review"].append(
                {
                    "shape_mnk": case["shape_mnk"],
                    "sides": sides,
                    "all_sides_pass_original_two_ulp_gate": passes_original,
                    "timing_status": "recorded errors also satisfy original two-ULP gate"
                    if passes_original
                    else "diagnostic only under weaker mixed-tolerance criterion",
                }
            )
            record["public_results"].append(
                {
                    key: case[key]
                    for key in (
                        "shape_mnk",
                        "seed",
                        "inputs_sha256",
                        "source_sha",
                        "source_diff_sha256",
                        "plan",
                        "compile_or_cache_lookup_seconds",
                        "memory_bytes",
                        "numerical_error_vs_independent_oracles",
                        "counts",
                        "baseline_scope",
                        "wall_clock_latency_ms",
                        "cuda_event_latency_ms",
                    )
                }
            )
    path = args.root / "P21-metrics-clean-graph-diagnostic" / "graph-diagnostic.json"
    graph = json.loads(path.read_text())
    if not graph["pass"] or len(graph["results"]) != 9 or len(record["public_results"]) != 9:
        raise ValueError("missing frozen or held-out shape")
    record["raw_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["graph_diagnostic"] = graph
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
