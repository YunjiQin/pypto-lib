# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build and replay one of the pinned TP8 LM-head cases in a fresh process."""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import golden
import golden.runner as runner
import torch
from pypto.ir import DistributedCompiledProgram
from pypto.runtime.debug.replay import invalidate_binary_cache
from pypto.runtime.distributed_runner import _assemble_chip_callables

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[2] / "models/deepseek_v4_1_flash"
PATCHES = HERE / "patches"
VARIANTS = json.loads((HERE / "variants.json").read_text())
FIXTURE = {
    "in/hidden_states.pt": "9fb7a855554c1b2cb6a2a3ab23fa2c21fd485bf674e53baaa303f07ac2f7dd68",
    "in/lm_head_weight.pt": "57f46380a98976622fc832953c7b21e7b5241fcb527e6abeacb64765dcd52622",
    "in/logit_row_indices.pt": "ebb7a93d14751fe0c45f7080096b5626665e23ff45defad05d01f2b941497381",
    "out/logits.pt": "b05a05518766fc9405d92dd42cf31677f70a1250b087212d1fa75df1331b05c6",
    "out/sampled_ids.pt": "4bab99adf735e69905ea54abec4be1ffb2aeb7d646dff9535f4ca0ef3b5b7b75",
}
BASELINE_OUTPUT = {
    "logits": "8096a28afaae7d48c8844a8d1fb5d8e30ae390bf489ed5a1f1738a9de905983f",
    "sampled_ids": FIXTURE["out/sampled_ids.pt"],
}


def verify_runtime_pin():
    from pypto.runtime.runtime_pin import runtime_pin_status

    expected = (HERE / "simpler.pin").read_text().strip()
    installed = runtime_pin_status().installed
    if installed != expected:
        raise RuntimeError(
            f"Expected Simpler {expected}, loaded {installed}. "
            "Build and select the runtime checkout specified in simpler.pin."
        )
    print(f"Simpler native binding: {installed}", flush=True)


def digest(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def model_call(entry, devices):
    # The model reads TP/DP and row count at import time. Use its actual CLI to
    # obtain the first (full LM-head) call, before its separate TP1 sample test.
    class FoundCall(Exception):
        pass

    def collect(**kwargs):
        raise FoundCall(kwargs)

    sys.path.insert(0, str(MODEL))
    argv = [
        str(entry),
        "-p",
        "a5",
        "--lm-head-tp",
        "8",
        "--dp",
        "1",
        "--logit-rows",
        "72",
        "--num-tokens",
        "72",
        "--enable-chip-swimlane",
        "4",
        "-d",
        devices,
    ]
    with patch.object(sys, "argv", argv), patch.object(golden, "run", collect):
        try:
            runpy.run_path(str(entry), run_name="__main__")
        except FoundCall as found:
            return found.args[0]
        finally:
            sys.path.pop(0)
    raise RuntimeError("LM-head entry did not issue a golden.run call")


def fixture_directory(out, existing):
    if existing:
        directory = existing.resolve()
    else:
        work = out / "fixture"
        work.mkdir()
        call = model_call(MODEL / "lm_head.py", "0,1,2,3,4,5,6,7")
        specs = call["specs"]
        for spec in specs:
            spec.direction = "out" if spec.name in BASELINE_OUTPUT else "in"
        torch.manual_seed(1854)
        tensors, scalars = runner._prepare_inputs(specs, specs, [], None, work)
        runner._compute_golden(specs, specs, scalars, tensors, work, None, call["golden_fn"])
        directory = work / "data"
    for name, expected in FIXTURE.items():
        tensor = torch.load(directory / name, weights_only=True, map_location="cpu", mmap=True)
        if digest(tensor) != expected:
            raise ValueError(f"{name}: fixture payload differs from the seed-1854 reference")
    (out / "fixture.path").write_text(str(directory) + "\n")


def apply_edit(directory, name):
    # The targets are scratch sources and JIT files, including when --out is
    # inside a worktree. Keep Git from discovering or editing a parent checkout.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(directory.parent))
    subprocess.run(
        ["git", "apply", "--no-index", "--ignore-space-change", str(PATCHES / f"{name}.patch")],
        cwd=directory,
        env=env,
        check=True,
    )


def prepare(path, variant):
    model = path / "model"
    model.mkdir(parents=True)
    entry = model / "lm_head.py"
    shutil.copyfile(MODEL / entry.name, entry)
    (model / "kernels").symlink_to(MODEL / "kernels", target_is_directory=True)
    model_edits = VARIANTS[variant]["model_edits"]
    native_edits = VARIANTS[variant]["chip_edits"]
    for name in ("barrier", *model_edits):
        apply_edit(model, name)
    os.chdir(path)
    call = model_call(entry, "0,1,2,3,4,5,6,7")
    result = golden.run(**(call | {"compile_only": True}))
    if not result.passed:
        raise RuntimeError(result.error)
    artifact = Path(result.work_dir).resolve()
    if "urma" in model_edits:
        apply_edit(artifact / "orchestration", "urma-domains")
    (chip,) = (artifact / "next_levels").glob("*/kernel_config.py")
    for name in native_edits:
        if name == "isa-submit":
            kernels = chip.parent / "kernels/aiv"
            shutil.copyfile(
                kernels / "lm_head_combine_notify.cpp", kernels / "lm_head_dispatch_notify_after_urma.cpp"
            )
        apply_edit(chip.parent, name)
    if native_edits:
        invalidate_binary_cache(artifact)
    _assemble_chip_callables(DistributedCompiledProgram.from_dir(artifact))
    (path / "artifact.path").write_text(str(artifact) + "\n")


def capture(path, devices):
    from simpler.task_interface import _native_set_host_log_directory

    ids = [int(i) for i in devices.split(",")]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError(f"Expected eight distinct device IDs, got {devices}")
    artifact = Path((path / "artifact.path").read_text().strip())
    archived = artifact / "host-logs"
    archived.mkdir()
    if not _native_set_host_log_directory(str(archived)):
        raise RuntimeError(f"Cannot open host timing log directory {archived}")
    os.chdir(path)
    call = model_call(path / "model/lm_head.py", devices)
    validate = runner._validate
    actual = {}

    def compare(specs, tensors, outputs, *args, **kwargs):
        actual.update({name: digest(tensors[name]) for name in outputs})
        return validate(specs, tensors, outputs, *args, **kwargs)

    with patch.object(runner, "_validate", compare):
        result = golden.run(
            **(
                call
                | {
                    "runtime_dir": str(artifact),
                    "golden_data": (path.parent / "fixture.path").read_text().strip(),
                }
            )
        )
    error = result.error or ""
    record = {
        "sha256": actual,
        "matches_hardware_baseline": actual == BASELINE_OUTPUT,
        "torch_golden_passed": result.passed,
        "torch_golden_error": error,
    }
    (path / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    known = "error_count=34/74465280" in error and "['logits']" in error
    if actual != BASELINE_OUTPUT or (not result.passed and not known):
        raise RuntimeError(f"Output validation differs from hardware reference: {record}")
    print("Hardware output hashes match. Strict Torch result:", "PASS" if result.passed else error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fixture", "prepare", "capture"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--device", default="0,1,2,3,4,5,6,7")
    a = parser.parse_args()
    out = a.out.resolve()
    if a.stage in ("prepare", "capture"):
        verify_runtime_pin()
    if a.stage == "fixture":
        fixture_directory(out, a.fixture)
    elif a.stage == "prepare":
        prepare(out / a.variant, a.variant)
    else:
        capture(out / a.variant, a.device)
