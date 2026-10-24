"""Inspect recorded evidence, recompute paired ratios, or rerun saved inputs.

The input and result ZIPs are read directly; no manual extraction is required.
Run `python reproduce.py --help` for commands. Fresh outputs are kept separate
from the immutable historical evidence.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
sys.path.insert(0, str(ROOT / 'code'))


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def index():
    return json.loads((DATA / 'INDEX.json').read_text())


def read_records(archive, stage):
    return [json.loads(line) for line in archive.read(stage + '.jsonl').decode().splitlines()
            if line.strip()]


def verify():
    """Check payload hashes and historical record/input bindings, not certificates."""
    manifest = json.loads((ROOT / 'CHECKSUMS.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'CHECKSUMS.json'
              and '__pycache__' not in p.parts and '.git' not in p.parts
              and '.DS_Store' != p.name and 'generated_runs' not in p.relative_to(ROOT).parts}
    check(set(manifest) <= actual, 'Missing distributed files')
    for name, expected in manifest.items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        check(digest == expected, f'Payload hash mismatch: {name}')
    spec = index()
    with ZipFile(DATA / 'inputs.zip') as archive:
        check(set(archive.namelist()) == set(spec['inputs']), 'Input ZIP membership mismatch')
        for name, expected in spec['inputs'].items():
            raw = archive.read(name)
            check(len(raw) == expected['bytes'], f'Input size mismatch: {name}')
            check(hashlib.sha256(raw).hexdigest() == expected['sha256'], f'Input hash mismatch: {name}')
    run_count = hash_bound = name_bound = 0
    with ZipFile(DATA / 'recorded_results.zip') as archive:
        check(archive.testzip() is None, 'Result ZIP CRC failure')
        for stage in json.loads((DATA / 'STAGE_INVENTORY.json').read_text()):
            rows = read_records(archive, stage['stage'])
            check(len(rows) == stage['records'], f'{stage["stage"]}: record count mismatch')
            keys = {(r['case_id'], r['method'], r['repeat']) for r in rows}
            check(len(keys) == len(rows), f'{stage["stage"]}: duplicate run key')
            check(len({r['case_id'] for r in rows}) == stage['inputs'], 'Input count mismatch')
            check(sorted({r['method'] for r in rows}) == sorted(stage['methods']), 'Method mismatch')
            check(sum(r['state'] == 'success' for r in rows) == stage['successes'], 'Success count mismatch')
            for r in rows:
                path = spec['case_inputs'][stage['stage']][r['case_id']]
                if r.get('instance_sha256'):
                    check(r['instance_sha256'] == spec['inputs'][path]['sha256'], 'Historical input binding mismatch')
                    hash_bound += 1
                else:
                    check(Path(path).stem == r['case_id'], 'Case-name binding mismatch')
                    name_bound += 1
            run_count += len(rows)
    print(json.dumps(dict(state='passed', payload_files=len(manifest),
                          saved_inputs=len(spec['inputs']), recorded_runs=run_count,
                          original_hash_bound_runs=hash_bound,
                          historical_case_name_bound_runs=name_bound)))


def cell(meta):
    return (meta['family'], int(meta['m']), float(meta.get('positive_fraction', .5)),
            float(meta.get('correlation', 0.)))


def paired_rows():
    """Median of three runs per input, then geometric mean of paired ratios."""
    spec = index()
    output = []
    with ZipFile(DATA / 'recorded_results.zip') as archive:
        for stage, comparators in sorted(spec['comparators'].items()):
            records = read_records(archive, stage)
            grouped = defaultdict(list)
            for r in records:
                grouped[r['case_id'], r['method']].append(r)
            valid = {}
            cells = {}
            methods = sorted({r['method'] for r in records})
            for (case_id, method), runs in grouped.items():
                cells[case_id] = cell(runs[0]['meta'])
                if len(runs) == 3 and {r['repeat'] for r in runs} == {0, 1, 2} and all(r['state'] == 'success' for r in runs):
                    times = [r['wall_total'] for r in runs]
                    check(all(math.isfinite(t) and t > 0 for t in times), 'Invalid timing')
                    valid[case_id, method] = statistics.median(times)
            for method in methods:
                for comparator in comparators:
                    ratios = defaultdict(list)
                    for case_id, group in cells.items():
                        a = valid.get((case_id, method))
                        b = valid.get((case_id, comparator))
                        if a is not None and b is not None:
                            ratios[group].append(math.log(a / b))
                    for group, logs in sorted(ratios.items()):
                        output.append(dict(stage=stage, family=group[0], m=group[1],
                            positive_fraction=group[2], mixing=group[3], method=method,
                            comparator=comparator, paired_inputs=len(logs),
                            time_ratio=math.exp(statistics.mean(logs))))
    return output


def row_key(row):
    return (row['stage'], row['family'], int(row['m']), float(row['positive_fraction']),
            float(row['mixing']), row['method'], row['comparator'])


def summarize(args):
    rows = paired_rows()
    if args.check:
        with (DATA / 'PAPER_PAIRED_SUMMARIES.csv').open(newline='') as handle:
            expected = {row_key(r): r for r in csv.DictReader(handle)}
        actual = {row_key(r): r for r in rows}
        check(actual.keys() == expected.keys(), 'Paired-comparison row membership changed')
        for key, row in actual.items():
            target = expected[key]
            check(row['paired_inputs'] == int(target['paired_inputs']), f'Pair count mismatch: {key}')
            check(math.isclose(row['time_ratio'], float(target['time_ratio']), rel_tol=1e-12, abs_tol=1e-14),
                  f'Ratio mismatch: {key}')
        print(f'PASS: {len(rows)} recorded paired comparisons independently recomputed.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        check(not args.output.exists(), 'Output already exists; choose a fresh path')
        with args.output.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {args.output}')
    if not args.check and not args.output:
        print('Paired cell ratios by stage/method/comparator:')
        groups = defaultdict(list)
        for row in rows:
            groups[row['stage'], row['method'], row['comparator']].append(row['time_ratio'])
        for (stage, method, comparator), values in sorted(groups.items()):
            ratio = math.exp(statistics.mean(math.log(v) for v in values))
            print(f'{stage:32} {method:24}/{comparator:24} {ratio:.6f} ({len(values)} successful cells)')
        print('These are equal-cell summaries; inspect failures and stage definitions before interpretation.')


def list_cases(args):
    spec = index()
    if not args.stage:
        for stage, cases in sorted(spec['case_inputs'].items()):
            print(stage, len(cases), 'inputs')
    else:
        for case_id, path in sorted(spec['case_inputs'][args.stage].items()):
            print(case_id, path)


def solve_saved(d, method, runner):
    """Map historical labels and retain the original basis-control behavior."""
    import engine
    import highspy
    from common_coefficient import run_q
    reverse = {value: key for key, value in runner.ALIASES.items()}
    method = reverse.get(method, method)
    if method in runner.METHODS + runner.CONTROLS:
        return runner.solve(d, method)
    if method not in ('N_without_basis', 'NP_same_ray', 'Q_without_basis'):
        raise ValueError(f'Unsupported historical method: {method}')
    old_lp, old_point = engine.LP, engine.Context.point

    class NoBasisLP(old_lp):
        def solve(self, *args, **kwargs):
            if self.solved:
                check(self.h.clearSolver() == highspy.HighsStatus.kOk, 'clearSolver failed')
                self.solved = False
            return super().solve(*args, **kwargs)

    def same_ray(context, raw, j=None):
        point = old_point(context, raw, j)
        producer = inspect.currentframe().f_back.f_locals.get('j')
        for t in range(len(context.d.b0)):
            if t != producer:
                point['qlo'][t] = -math.inf
                point['bhi'][t] = math.inf
                point['blo'][t] = -math.inf
        return point

    try:
        if method in ('N_without_basis', 'Q_without_basis'):
            engine.LP = NoBasisLP
        if method == 'NP_same_ray':
            engine.Context.point = same_ray
        result = run_q(d) if method == 'Q_without_basis' else engine.run_method(d, 'NP' if method == 'NP_same_ray' else 'N')
    finally:
        engine.LP, engine.Context.point = old_lp, old_point
    result['method'] = method
    return result


def rerun(args):
    import numpy as np
    import scipy
    import highspy
    import engine
    import run as runner
    spec = index()
    cases = sorted(spec['case_inputs'][args.stage])
    if args.case:
        check(args.case in cases, 'Case does not belong to selected stage')
        cases = [args.case]
    elif args.limit:
        cases = cases[:args.limit]
    with ZipFile(DATA / 'recorded_results.zip') as archive:
        historical = read_records(archive, args.stage)
    available = sorted({r['method'] for r in historical})
    reverse = {value: key for key, value in runner.ALIASES.items()}
    requested = args.methods or available
    selected = [m for m in available if m in requested or reverse.get(m, m) in requested]
    check(selected and all(m in available or any(reverse.get(a, a) == m for a in available) for m in requested),
          f'Select methods present in this stage: {available}')
    check(not args.output.exists(), 'Fresh-run output directory already exists')
    args.output.mkdir(parents=True)
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, scipy=scipy.__version__,
                    highspy=highspy.Highs().version(), stage=args.stage,
                    cases=cases, methods=selected, repeats=args.repeats,
                    code_sha256={p.name: engine.fingerprint(p) for p in (ROOT / 'code').glob('*.py')},
                    runner_sha256=engine.fingerprint(__file__),
                    timing_scope='new serial rerun; method-local timing; fresh state per method',
                    method_order='seed + repeat * 100003; seeded permutation; may differ from historical stage')
    (args.output / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    failures = 0
    with ZipFile(DATA / 'inputs.zip') as archive:
        for case_id in cases:
            path = spec['case_inputs'][args.stage][case_id]
            d = engine.load_instance(io.BytesIO(archive.read(path)))
            for repeat in range(args.repeats):
                for method in np.random.default_rng(d.meta['seed'] + repeat * 100003).permutation(selected):
                    method = str(method)
                    result = solve_saved(d, method, runner)
                    for key in [key for key in result if key.startswith('_')]:
                        arrays = result.pop(key)
                        if args.save_artifacts and isinstance(arrays, dict):
                            np.savez_compressed(args.output / f'{case_id}_{method}_r{repeat}{key}.npz', **arrays)
                    result.update(method=method, case_id=case_id, repeat=repeat, stage=args.stage,
                                  input_member=path, instance_sha256=spec['inputs'][path]['sha256'],
                                  record_kind='new rerun; not a historical timing')
                    with (args.output / 'results.jsonl').open('a') as handle:
                        handle.write(json.dumps(result, default=runner.json_value, allow_nan=False) + '\n')
                    print(case_id, repeat, method, result['state'], round(result['wall_total'], 5), flush=True)
                    if result['state'] != 'success':
                        print(result.get('failure_reason', ''), flush=True)
                        failures += 1
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='verify distributed file hashes, records, and input bindings')
    show = sub.add_parser('list', help='list stages or saved case IDs')
    show.add_argument('--stage', choices=sorted(index()['case_inputs']))
    summary = sub.add_parser('summarize', help='recompute historical paired geometric time ratios')
    summary.add_argument('--check', action='store_true', help='compare all ratios with original CSV')
    summary.add_argument('--output', type=Path, help='write recomputed cell comparisons to a new CSV')
    fresh = sub.add_parser('rerun', help='run methods on original binary64 inputs with fresh solver state')
    fresh.add_argument('--stage', required=True, choices=sorted(index()['case_inputs']))
    fresh.add_argument('--case', help='one exact case ID from the list command')
    fresh.add_argument('--limit', type=int, default=1, help='cases in sorted order; 0 selects all')
    fresh.add_argument('--methods', nargs='+', help='historical or manuscript method labels')
    fresh.add_argument('--repeats', type=int, default=3)
    fresh.add_argument('--save-artifacts', action='store_true', help='save NEW proof/pool arrays locally')
    fresh.add_argument('--output', type=Path, required=True, help='new directory for fresh outputs')
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
    elif args.command == 'list':
        list_cases(args)
    elif args.command == 'summarize':
        summarize(args)
    else:
        if args.limit < 0 or args.repeats <= 0:
            parser.error('limit must be nonnegative and repeats must be positive')
        return rerun(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
