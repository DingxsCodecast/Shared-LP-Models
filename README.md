# Shared LP Models for Signed Polar-Cut Computations

Code and compact experimental evidence for **Shared LP Models for Signed
Polar-Cut Computations in Disjoint Bilinear Programming**, prepared for
*Optimization Letters*.

This repository includes runnable algorithms, the original binary64 input
instances, and the original run records needed to recompute the archived paired
time ratios. Large historical point pools and proof arrays are omitted.

## Contents

| Path | Purpose |
| --- | --- |
| `code/` | Algorithms, instance generators, and a standalone runner |
| `reproduce.py` | Verify the package, recompute recorded ratios, and rerun saved inputs |
| `requirements.txt` | The three runtime dependencies, with tested versions |
| `data/inputs.zip` | 228 original input instances, approximately 4.1 MB |
| `data/recorded_results.zip` | 6,921 original run records across 23 stages, stage metadata, and the unresolved-generation ledger; approximately 5.2 MB |
| `data/PAPER_PAIRED_SUMMARIES.csv` | Original 1,410 paired cell comparisons |
| `data/STAGE_INVENTORY.json` | Stage counts, methods, successes, and failures |
| `data/INDEX.json` | Input hashes, stage/case mappings, and declared comparators |
| `CHECKSUMS.json` | SHA-256 hashes for the distributed files |

The data ZIPs are read directly by `reproduce.py`. They should remain zipped
in GitHub; no manual extraction is needed. A download of the repository contains
all the data required for the commands below, without Git LFS or an external
data service.

## Quick start

Use **Python 3.12 on macOS or Linux**. The examples use `python` to mean the
interpreter for that environment. The implementation uses the POSIX `resource`
module; Windows users can use a Linux environment such as WSL.

```bash
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py summarize --check
python code/run.py --self-check
```

`verify` checks file hashes, stage counts, unique run keys, and input bindings.
It does **not** re-audit omitted historical proof arrays. There are 6,849
historical records with an original input hash; the 72 older `basis` records
have only case IDs and are matched by those IDs. The distributed input files
are independently hashed in `INDEX.json`.

`summarize --check` recomputes all 1,410 recorded paired comparisons from the
raw run records and checks the original CSV to floating-point tolerance. It
does not obtain those ratios by copying the original CSV.

`--self-check` checks all 17 main method implementations and four supporting
controls against exact vertex optima on small mixed-sign instances. It also
checks RN/RD proof equality and rejection of unavailable positive margins.

## Recompute recorded comparisons

```bash
python reproduce.py summarize --check --output generated_runs/paired_comparisons.csv
python reproduce.py list
python reproduce.py list --stage strong
```

For each input/method, the calculation takes the median of three successful
fresh-state repetitions, then geometrically averages the paired relative
times within each predefined cell. Both methods must succeed in all three
repetitions. Failures are retained in the raw records and stage inventory.
For the paper's family aggregates, geometrically average the applicable cell
ratios with equal cell weights. Later runs on the same geometry do not count
as independent instances, and times across different stages are not paired
comparisons.

## Run algorithms on the original saved instances

```bash
python reproduce.py rerun --stage strong --limit 1 --methods NPSN QP QD --repeats 3 --output generated_runs/strong_example
python reproduce.py rerun --stage memo20261005_confirm_main --limit 1 --methods RN RD NPSN QD --repeats 3 --output generated_runs/rd_example
```

The default is the first saved case in sorted order. Use `--case` with an ID
from `list`, or `--limit 0` to select every case in a stage. Omitting `--methods`
selects the methods recorded for that stage. Historical implementation labels
and the manuscript aliases are accepted. Each method starts from fresh solver
state. Inputs are taken from the saved ZIP, not regenerated from nominal seeds.

New timings and environment information go into the chosen new output folder.
The folder must not already exist. Optional `--save-artifacts` also saves newly
computed proof/pool arrays there. These outputs are ignored by Git when placed
under `generated_runs/`. A failed or unresolved selected method yields a
nonzero exit status, rather than silently removing that observation.

Some historical scale and small-slope inputs intentionally have timeouts or
unresolved results. Rerunning them can be expensive. Hardware, platform, solver
versions, and the order of method execution can affect timings; a fresh rerun
is not expected to reproduce historical wall-clock seconds exactly.

## Generate fresh instances

```bash
python code/run.py
python code/run.py --workload within-box --methods NPSN QP QD RN RD
python code/run.py --family bounded_sparse_coupled --workload related --m 128 --k 32 --correlation 0.9 --methods NPSN QP QD RN RD
```

The default generates one F1 and one F2 instance and runs the 17 main methods.
The within-box generator retains its original fixed dimensions, m=128 and
k=16. Generation is outside the method timer and gives no free generation
witnesses to timed methods. These commands are demonstrations with fresh
inputs, not replacements for the original experimental evidence.

## Scope and provenance

Only the positive-origin-margin signed polar-cut inner module is implemented;
this is not a complete outer DBLP solver. The numerical algorithm/generator
functions and classes are retained from the original evidence archive.
Historical campaign entry points that required internal plan files have been
replaced by standalone entry points.

Original NPZ inputs, JSONL records, available stage metadata, the paired CSV,
and the stage inventory are preserved byte-for-byte. Historical metadata may
refer to original script hashes, internal plan files, point pools, and proof
files that are not distributed here. Those entries are provenance, not current
file dependencies. Hashes for this compact repository are in `CHECKSUMS.json`.

The omitted arrays are unnecessary for recomputing the recorded timing ratios
or running the algorithms. The algorithms still perform their numerical
certification. However, this compact repository cannot independently reproduce
the historical exact certificate audits or check historical RN/RD proof-array
equality. `--self-check` and `--save-artifacts` concern newly computed proofs.

The original source-archive hash is recorded in `data/INDEX.json`.
