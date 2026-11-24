# Reproduce the eight TP8 LM-head push variants

This experiment runs the full DeepSeek V4.1 Flash LM-head from pypto-lib PR
#1854 on eight Ascend 950DT devices. Each owner selects 72 rows of 5120 BF16
values and sends 720 KiB to each of seven peers. An eight-rank barrier
precedes row selection. The projection, logits exchange and greedy sampling
run after the hidden-state gather.

## Environment

Use Linux, Git, Python 3.11, CPU PyTorch 2.8.0, the A5 onboard Simpler
runtime, CANN 9.3.0, PTOAS 0.72 and a working eight-card communication
domain. The reference machine used driver 25.6.rc3.1. Start in this
pypto-lib checkout with its Python dependencies and CANN environment active.
The base pypto-lib commit is `461f1a6154f77afa346818596731bade5f7988be`.

The compiler uses the PyPTO integration snapshot. The runtime uses a fixed
commit from Simpler main, recorded in [simpler.pin](../../examples/experiments/tp8_lm_head/simpler.pin).
From the pypto-lib root, select that runtime after initializing the PyPTO
submodules and before building or installing either package.

```bash
git clone --branch integration/urma-ccu-test https://github.com/hw-native-sys/pypto.git ../pypto-tp8
git -C ../pypto-tp8 checkout --detach 864c8b86b2bae5c53b951d880b6794c5887cb0d2
git -C ../pypto-tp8 submodule update --init --recursive
export PYPTO_ROOT="$(cd ../pypto-tp8 && pwd)"
simpler_pin=$(cat examples/experiments/tp8_lm_head/simpler.pin)
git -C "$PYPTO_ROOT/runtime" fetch origin "$simpler_pin"
git -C "$PYPTO_ROOT/runtime" checkout --detach "$simpler_pin"
cat "$PYPTO_ROOT/runtime/pto_isa.pin"
```

This Simpler revision pins PTO-ISA `45435c0624c1d0d056bf38aa60d4f56dc0ef3b6b`.
PyPTO can warn that its original `runtime/` gitlink differs from the working
tree. The runtime checkout and the installed native binding must both match
the experiment pin; the runner checks the binding before compiling or capturing. Install PyPTO and Simpler for the
`a5` target using the [installation guide](../get-started/installation.md).
An existing built checkout at these revisions works too: use its Python
environment, or add its `python`, `runtime` and `runtime/python` directories to
`PYTHONPATH` when using an in-tree build. Set `PTOAS_ROOT`
to the directory containing `bin/ptoas` and verify the version with
`"$PTOAS_ROOT/bin/ptoas" --version`.

## Run

From the pypto-lib root, run all eight variants with one command:

```bash
python examples/experiments/tp8_lm_head/reproduce.py run --out build_output/tp8-all
```

On a host with `task-submit`, the runner generates or verifies the fixture and
builds the kernels off-device, queues the captures together, waits, then
merges the timing passes. An existing queue allocation supplies `TASK_DEVICE`. To request specific
cards, use `--device 0,1,2,3,4,5,6,7`. Without a queue, that option selects
the devices allocated to your process. Use a fresh output directory
for each run. On a queue host the job must see the same checkout, output
directory, toolchain and Python executable.

For a subset, or to perform compilation and hardware capture separately:

```bash
script=examples/experiments/tp8_lm_head/reproduce.py
out=build_output/tp8-mte-pair
python "$script" prepare --out "$out" --variants mte-original mte-pipeline
python "$script" capture --out "$out" --variants mte-original mte-pipeline
python "$script" analyze --out "$out" --variants mte-original mte-pipeline
```

`--fixture /path/to/data` on `prepare` or `run` reuses a directory with `in/`
and `out/`. Without it the script generates the approximately 1.6 GB CPU
fixture using seed 1854. Both routes verify the hashes of all five tensor
payloads, including the Torch reference outputs.

## Variants and reference captures

All times below are in microseconds. The rank maximum estimates simultaneous
block start by taking the longest push block per rank; the table reports the
median of those eight maxima. Worker span includes the actual stagger between
the first block starting and the last block ending. Scheduler latency starts
at the first push dispatch. Its endpoint is the last scheduler-observed worker
FIN for synchronous transfers and the successful scheduler poll for deferred
tasks. Times use rank-local clocks and exclude the separate notify/wait tasks.

| CLI variant | Remote push implementation | Median rank max | Max across 8 cards | Median worker span | Median scheduler latency |
|---|---|---:|---:|---:|---:|
| `mte-original` | 64 AIV blocks, synchronous MTE row tiles | 27.97 | 31.11 | 33.48 | 52.54 |
| `mte-pipeline` | One block per peer, two 32 KiB MTE staging buffers | 21.10 | 21.88 | 21.64 | 25.39 |
| `urma-checked` | Original PyPTO checks and Simpler submit/defer | 14.26 | 17.57 | 14.39 | 45.87 |
| `direct-wqe` | Fixed shape, one WRITE WQE per peer, manually register the Simpler deferred event | 6.67 | 6.92 | 6.92 | 47.11 |
| `isa-wait` | ISA session and TPUT, wait for the event in the sending block | 33.26 | 34.77 | 33.54 | 37.97 |
| `isa-submit` | ISA session and TPUT, return after submission | 9.40 | 10.10 | 9.64 | — |
| `simpler-defer` | Remove PyPTO checks, retain Simpler submission, validation and deferred completion | 11.32 | 12.22 | 11.76 | 45.97 |
| `ccu` | CCU allgather submitted by one AIV wrapper | 7.89 | 8.53 | 7.89 | 126.94 |

All eight variants run in one allocation. Per-peer variants use seven remote
push blocks and 24 local three-row copies. In the five URMA variants, the
registered hidden payload has its own URMA domain; the MTE start barrier,
signals and logits exchange share a peer-mapped device domain. Simpler main
leaves peer VA mappings out of URMA domains. In `isa-submit` the dependent
hidden-state notifier drains the posted queue using the payload context,
then signals the peer using the MTE context.
Its push bar measures submission only, so the scheduler transfer-completion
column is empty. The CCU wrapper bar also measures submission. The
`direct-wqe` patch assumes this fixture's single outstanding WRITE per queue
and one fresh completion slot per remote block.

## Validation and artifacts

Each capture checks device logits and sampled IDs against the recorded
hardware output hashes, and also runs the original Torch comparison. On this
fixture, sampled IDs pass and 34 of 74,465,280 logits exceed the strict
`atol=rtol=0.001` threshold. The runner records the strict result even when
the outputs match the hardware baseline. A hardware digest mismatch, a new
comparison failure, an incomplete trace or an unexpected completion path
fails the reproduction.

- `summary.csv` contains the four timing columns above for the selected variants.
- `<variant>.perfetto.json` is the uncompressed eight-rank timing trace.
- `<variant>/ranks.csv`, `validation.json` and `artifact.path` give per-rank
  metrics, output checks and the path to the generated C++ and raw traces.
- `<variant>-prepare.log` and `<variant>-capture.log` retain the compiler and
  device logs. The fixture, model-source overlays and generated binaries live
  under the chosen output directory.

The [runner](../../examples/experiments/tp8_lm_head/reproduce.py) selects edits via the
[variant manifest](../../examples/experiments/tp8_lm_head/variants.json).
Model patches reproduce the original DAG and barrier. The URMA host edit
separates payload and signal domains. Native patches are applied to freshly
generated chip sources before assembly and replay.
