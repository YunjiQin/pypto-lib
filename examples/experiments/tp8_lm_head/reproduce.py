# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproduce the full eight-rank A5 LM-head push comparison."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VARIANTS = json.loads((HERE / "variants.json").read_text())


def case(out, args, log_name):
    log = out / log_name
    print(f"{' '.join(args)} -> {log}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            [sys.executable, HERE / "case.py", args[0], out, *args[1:]],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        print("\n".join(log.read_text(errors="replace").splitlines()[-25:]), file=sys.stderr)
        raise SystemExit(result.returncode)


def prepare(args, out):
    out.mkdir(parents=True, exist_ok=True)
    if any((out / v).exists() for v in args.variants) or (out / "fixture.path").exists():
        raise SystemExit(
            f"Preparation already started in {out}. Choose a fresh --out or use capture/analyze."
        )
    fixture_args = ["--fixture", str(args.fixture.resolve())] if args.fixture else []
    case(out, ["fixture", *fixture_args], "fixture.log")
    for variant in args.variants:
        case(out, ["prepare", "--variant", variant], variant + "-prepare.log")


def capture(args, out):
    devices = os.environ.get("TASK_DEVICE")
    queue = shutil.which("task-submit")
    if not queue:
        devices = devices or args.device
    if devices:
        for v in args.variants:
            case(out, ["capture", "--variant", v, "--device", devices], v + "-capture.log")
        return
    if not queue:
        raise SystemExit("Supply --device with the eight allocated IDs, or run on a host with task-submit.")
    cmd = [
        sys.executable,
        str(HERE / "reproduce.py"),
        "capture",
        "--out",
        str(out),
        "--variants",
        *args.variants,
    ]
    allocation = ["--device", args.device] if args.device else ["--device", "auto", "--device-num", "8"]
    submit = [queue, *allocation, "--timeout", "3600", "--max-time", "3600"]
    for name in ("PYPTO_ROOT", "PTOAS_ROOT", "PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "PTO_ISA_ROOT"):
        if os.environ.get(name):
            submit.extend(["--env", name])
    reply = subprocess.check_output(submit + [shlex.join(cmd) + ' --device "$TASK_DEVICE"'], text=True)
    job = re.search(r"\btask_[\w-]+", reply)
    if not job:
        raise RuntimeError(f"Missing task ID in queue reply: {reply}")
    print(f"Eight-card job {job[0]}", flush=True)
    subprocess.run(["task-submit", "--timeout", "3600", "--wait", job[0]], check=True)
    for v in args.variants:
        if not (out / v / "validation.json").is_file():
            raise RuntimeError(f"{v}: missing device validation after job {job[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", choices=("run", "prepare", "capture", "analyze"), default="run")
    parser.add_argument("--out", type=Path, help="Persistent run directory; required for capture/analyze")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--fixture", type=Path, help="Reuse the seed-1854 data directory (in/ and out/)")
    parser.add_argument("--device", help="Eight already allocated, comma-separated device IDs")
    args = parser.parse_args()
    if args.out is None:
        if args.phase not in ("run", "prepare"):
            parser.error("pass the prepared --out directory")
        artifacts = REPO / "build_output"
        artifacts.mkdir(exist_ok=True)
        args.out = Path(tempfile.mkdtemp(prefix="tp8-push-", dir=artifacts))
    out = args.out.resolve()
    print(f"Run directory: {out}", flush=True)
    # The compile, queue and analysis subprocesses must import this harness.
    paths = [str(REPO)]
    paths.extend(filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)))
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    os.environ.update(
        PYPTO_BENCH="0",
        PYPTO_RUNTIME_LOG="timing",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        PYPTO_GOLDEN_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    if args.phase in ("run", "prepare"):
        prepare(args, out)
    if args.phase in ("run", "capture"):
        capture(args, out)
    if args.phase in ("run", "analyze"):
        subprocess.run([sys.executable, HERE / "metrics.py", out, *args.variants], check=True)


if __name__ == "__main__":
    main()
