# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Measure rank-local push durations and export uncompressed multi-rank Perfetto traces."""

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def measure_rank(directory, variant, rank):
    identity = read(directory / "dispatch_identity.timing.json")
    raw = read(directory / "chip_swimlane_records.json")
    if identity["chip_rank"] != rank or raw["chip_swimlane_level"] != 4:
        raise ValueError(f"{directory}: unexpected rank or capture level")
    graph = read(directory / "deps.json")
    names = read(directory / "name_map.json")["callable_id_to_name"]
    tasks = sorted(
        (int(t["task_id"]), names[str(max(t["kernel_ids"]))], t["block_num"])
        for t in graph["tasks"]
        if max(t["kernel_ids"]) >= 0
    )
    barrier, select, push = tasks[:3]
    expected_blocks = {"mte-original": 64, "ccu": 1}.get(variant, 31)
    expected_name = "ccu" if variant == "ccu" else "dispatch_push"
    if (
        barrier[1] != "tp8_start_barrier"
        or "select_rows" not in select[1]
        or expected_name not in push[1]
        or push[2] != expected_blocks
    ):
        raise ValueError(f"{variant}: unexpected push graph {tasks[:3]}")
    edges = {(int(e["pred"]), int(e["succ"])) for e in graph["edges"] if "wait" in e["flags"]}
    if not {(barrier[0], select[0]), (select[0], push[0])} <= edges:
        raise ValueError("Missing barrier -> select -> push dependency")

    sched = {
        (epoch, core, reg): (start, end) for core, reg, start, end, epoch in raw["scheduler_tasks"]["records"]
    }
    intervals = [
        (start, end, *sched[epoch, core, reg], epoch)
        for core, token, reg, start, end, _, epoch in raw["aicore_tasks"]
        if token == push[0]
    ]
    if len(intervals) != expected_blocks:
        raise ValueError(f"Missing push blocks: expected {expected_blocks}, got {len(intervals)}")
    starts, ends, dispatches, finishes, epochs = zip(*intervals, strict=True)
    (epoch,) = set(epochs)
    streams = raw["scheduler_records"]["streams"]
    if any(s["capture"]["truncated"] or s["capture"]["dropped"] for s in streams):
        raise ValueError(f"{directory}: incomplete scheduler stream")
    polls = [
        p
        for s in streams
        for p in s["records"]
        if p["kind"] == "async_poll" and p["run_epoch"] == epoch and p["tasks_processed"] > 0
    ]
    deferred = variant in {"ccu", "urma-checked", "simpler-defer", "direct-wqe"}
    if sum(p["tasks_processed"] for p in polls) != int(deferred):
        raise ValueError(f"{variant}: unexpected deferred completion count")
    cycles_per_us = raw["metadata"]["clock_freq_hz"] / 1e6
    done = polls[0]["end_cycles"] if deferred else max(finishes)
    slowest = max(b - a for a, b in zip(starts, ends, strict=True))
    # ISA submit-only retires the WRITE in the downstream notifier. Worker FIN
    # marks submission completion and cannot stand in for transfer completion.
    row = dict(
        variant=variant,
        rank=rank,
        push_blocks=len(intervals),
        slowest_block_us=slowest / cycles_per_us,
        worker_span_us=(max(ends) - min(starts)) / cycles_per_us,
        scheduler_done_us=(done - min(dispatches)) / cycles_per_us if variant != "isa-submit" else None,
    )
    return row, identity["local_capture_index"]


def analyze(out, variant):
    path = out / variant
    if not read(path / "validation.json")["matches_hardware_baseline"]:
        raise ValueError(f"{variant}: hardware output digest mismatch")
    artifact = Path((path / "artifact.path").read_text().strip())
    rows = []
    indices = set()
    for rank in range(8):
        # Level 4 writes the graph and clean timing passes into the same d0.
        directory = artifact / f"dfx_outputs/rank{rank}/d0"
        row, index = measure_rank(directory, variant, rank)
        rows.append(row)
        indices.add(index)
        timing = path / "timing-pass" / f"rank{rank}/d{index}"
        timing.mkdir(parents=True, exist_ok=True)
        for name in ("deps", "name_map", "chip_swimlane_records", "dispatch_identity", "dispatch_program"):
            src = directory / (name + (".timing.json" if name.startswith("dispatch_") else ".json"))
            dest = timing / f"{name}.json"
            if not dest.exists():
                dest.symlink_to(src.resolve(strict=True))
    (index,) = indices
    trace = out / f"{variant}.perfetto.json"
    cmd = [
        sys.executable,
        "-m",
        "simpler_setup.tools.swimlane_converter",
        str(path / "timing-pass"),
        "--dispatch",
        f"d{index}",
        "-o",
        str(trace),
    ]
    for log in sorted((artifact / "host-logs").glob("host.*.log")):
        cmd.extend(["--host-log", str(log)])
    with (path / "merge.log").open("w") as stream:
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    metadata = read(trace)["metadata"]
    if metadata["rank_count"] != 8 or metadata["trace_status"] != "complete":
        raise ValueError(f"{trace}: incomplete multi-rank trace")
    write_csv(path / "ranks.csv", rows)
    latency = [r["scheduler_done_us"] for r in rows if r["scheduler_done_us"] is not None]
    return dict(
        variant=variant,
        median_rank_max_us=statistics.median(r["slowest_block_us"] for r in rows),
        max_block_us=max(r["slowest_block_us"] for r in rows),
        median_worker_span_us=statistics.median(r["worker_span_us"] for r in rows),
        median_scheduler_done_us=statistics.median(latency) if latency else None,
    )


if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    results = [analyze(out, variant) for variant in sys.argv[2:]]
    write_csv(out / "summary.csv", results)
    print((out / "summary.csv").read_text(), end="")
