"""Standalone code for the OPTL signed polar-cut computation module.

Environment: Python 3.12 on macOS/Linux. Install the three dependencies with:
    python -m pip install numpy==2.4.3 scipy==1.17.1 highspy==1.15.1

From this directory:
    python run.py --self-check
    python run.py
    python run.py --family transportation --m 128 --k 16 --repeats 3
    python run.py --workload within-box --methods NPSN QP QD RN RD
    python run.py --workload related --m 128 --k 32 --correlation 0.9
    python run.py --methods QP QD --output fresh_run
    python run.py --input fresh_run/input_bounded_sparse_coupled.npz

The default run generates fresh fixed-seed inputs for both F1 and F2 and runs
all 17 methods in the manuscript. No saved inputs or historical results are
required. --output optionally saves NEW inputs, results and proof arrays.
Generation is outside method timers; each repetition creates fresh model state.
Methods run serially because solver adapters temporarily change engine.LP.

Method labels follow the manuscript. N/D24/DL/NP are the original controls;
NS/NN/NSN/NPN/NPSN are strengthened Newton variants; Q/QP/QD are shared-model
methods; NC is classification-free; PC/GD/RN/RD refine Newton. Optional
SP/SG/PB/GR are supporting point/basis/bracket controls.

This code directory contains only Python source files. Original binary64
inputs and recorded timings are supplied separately in the repository data/.
Regenerated inputs and new timings are not the original experimental record.
The scope is the positive-origin-margin inner module, not an outer DBLP solver.
Small-slope or difficult instances can explicitly return unresolved outcomes;
the program exits unsuccessfully when a selected method does not succeed.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

import highspy
import numpy as np
import scipy
from scipy import sparse

import engine
from coefficient_newton import run_nc
from cutpoint_memo import run_cached
from negative_reuse import run_structure as run_residual
from strong_baseline import run as run_strong
from structure_newton_margin import run_structure as run_point
from structure_reuse import run_structure as run_pool
from within_box import generate as generate_box


METHODS = ('N', 'D24', 'DL', 'NP', 'NS', 'NN', 'NSN', 'NPN', 'NPSN',
           'Q', 'QP', 'QD', 'NC', 'PC', 'GD', 'RN', 'RD')
CONTROLS = ('SP', 'SG', 'PB', 'GR')
ALIASES = {'NS': 'N_saved_basis', 'NN': 'N_norm', 'NSN': 'N_saved_norm',
           'NPN': 'N_primal_norm', 'NPSN': 'N_primal_saved_norm', 'QP': 'Q_primal'}
FAMILIES = ('bounded_sparse_coupled', 'transportation')


def solve(instance, method):
    """Dispatch manuscript labels without changing numerical implementations."""
    if method in ('N', 'D24', 'DL', 'NP'):
        result = engine.run_method(instance, method)
    elif method in ALIASES or method in ('Q', 'QD'):
        result = run_strong(instance, ALIASES.get(method, method))
    elif method == 'NC':
        result = run_nc(instance)
    elif method in ('PC', 'PB', 'GR', 'GD'):
        result = run_pool(instance, method)
    elif method in ('SP', 'SG'):
        result = run_point(instance, method)
    elif method == 'RN':
        result = run_residual(instance, method)
    elif method == 'RD':
        result = run_cached(instance, method)
    else:
        raise ValueError(f'Unknown method: {method}')
    result['method'] = method
    return result


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def self_check():
    """Check against exact vertex optima, not another floating-point solver."""
    checks = pairs = rejected = 0
    for seed, m in enumerate((3, 5, 8, 12)):
        rng = np.random.default_rng(6180 + seed)
        a = rng.integers(0, 10, m).astype(float)
        B = rng.integers(-8, 9, (6, m)).astype(float)
        b0 = -B.min(axis=1) + np.array([-1., 1., -2., 2., -3., 3.])
        G = sparse.csr_matrix(np.vstack([np.ones(m), -np.ones(m)]))
        d = engine.Instance(G, np.array([1., -1.]), np.ones(m), a, 1.,
                            B, b0, np.full(m, 1 / m),
                            dict(family='simplex', m=m, k=6, seed=seed))
        truth = [max(-(Fraction(float(B[j, i])) + Fraction(float(b0[j]))) /
                     (Fraction(float(a[i])) + Fraction(float(d.a0)))
                     for i in range(m)) for j in range(6)]
        results = {}
        for method in METHODS + CONTROLS:
            result = solve(d, method)
            require(result['state'] == 'success',
                    f'{method}: {result.get("failure_reason", result["state"])}')
            require(len(result['rays']) == 6, f'{method}: missing rays')
            for ray in result['rays']:
                j = ray['j']
                lower, upper = Fraction(ray['q_lower']), Fraction(ray['q_safe'])
                require(lower <= truth[j] <= upper, f'{method}/{j}: bad enclosure')
                require(upper - lower <= Fraction(engine.TARGET * (1 + abs(ray['q_safe']))),
                        f'{method}/{j}: coefficient quality failed')
                require(ray['branch'] == ('positive' if truth[j] > 0 else 'negative'),
                        f'{method}/{j}: wrong signed branch')
                checks += 1
            results[method] = result
        rn, rd = results['RN'], results['RD']
        require(rn['lp_calls'] == rd['lp_calls'] and
                rn['simplex_iterations'] == rd['simplex_iterations'],
                'RN/RD changed the solve path')
        for key in ('q_safe', 'q_lower', 'cert_q_nom', 'cert_dual', 'lower_witness'):
            require(np.array_equal(rn['_proof'][key], rd['_proof'][key]),
                    f'RN/RD proof mismatch: {key}')
        pairs += 1
        d.a = np.zeros(m)
        d.a0 = 0.
        for method in ('N', 'QP', 'QD', 'NC', 'RN', 'RD'):
            result = solve(d, method)
            require(result['state'] == 'failed', f'{method}: accepted zero origin margin')
            rejected += 1
    print(json.dumps(dict(state='passed', methods=len(METHODS + CONTROLS),
                          exact_ray_checks=checks, identical_rn_rd_pairs=pairs,
                          invalid_margin_rejections=rejected)), flush=True)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--family', choices=('all',) + FAMILIES, default='all')
    parser.add_argument('--workload', choices=('main', 'within-box', 'related'), default='main',
                        help='within-box uses its original fixed m=128, k=16 generator')
    parser.add_argument('--m', type=positive_int, default=64, help='dimension (except within-box)')
    parser.add_argument('--k', type=positive_int, default=4, help='directions (except within-box)')
    parser.add_argument('--rho', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--correlation', type=float, default=0.)
    parser.add_argument('--beta-margin', type=float, default=1.)
    parser.add_argument('--repeats', type=positive_int, default=1)
    parser.add_argument('--methods', nargs='+', choices=METHODS + CONTROLS, default=list(METHODS))
    parser.add_argument('--input', type=Path, help='load an engine-format NPZ instead of generating')
    parser.add_argument('--output', type=Path, help='new empty directory for optional run outputs')
    args = parser.parse_args()
    if not (0 <= args.rho <= 1 and 0 <= args.correlation <= 1 and args.beta_margin > 0):
        parser.error('rho/correlation must be in [0,1]; beta-margin must be positive')
    if len(args.methods) != len(set(args.methods)):
        parser.error('methods must not contain duplicates')
    if args.self_check:
        self_check()
        return 0
    families = FAMILIES if args.family == 'all' else (args.family,)
    if not args.input and args.workload != 'within-box' and 'transportation' in families:
        p = 8 if args.m in (64, 128) else 16
        if args.m % p:
            parser.error(f'transportation with m={args.m} requires m divisible by {p}')
    if args.output:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            parser.error('--output must name a new or empty directory')
        args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(python=platform.python_version(), numpy=np.__version__,
                       scipy=scipy.__version__, highspy=highspy.Highs().version())
    print('Environment: ' + json.dumps(environment), flush=True)
    if args.output:
        source_hashes = {p.name: engine.fingerprint(p) for p in Path(__file__).parent.glob('*.py')}
        (args.output / 'environment.json').write_text(json.dumps(
            dict(environment=environment, code_sha256=source_hashes,
                 arguments={k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()}), indent=2) + '\n')
    if args.input:
        instances = [(args.input.stem, engine.load_instance(args.input))]
    else:
        instances = []
        for family in families:
            if args.workload == 'within-box':
                d, _ = generate_box(family, args.rho, args.seed)
            else:
                d = engine.generate(family, args.m, args.k, args.rho, args.seed,
                                    args.correlation, args.beta_margin)
            instances.append(('input_' + family, d))
    failures = 0
    for case, d in instances:
        if args.output:
            engine.save_instance(d, args.output / (case + '.npz'))
        for repeat in range(args.repeats):
            for method in np.random.default_rng(d.meta.get('seed', args.seed) + repeat * 100003).permutation(args.methods):
                method = str(method)
                result = solve(d, method)
                private = {key: result.pop(key) for key in list(result) if key.startswith('_')}
                if result['state'] == 'success':
                    require(len(result['rays']) == len(d.b0), f'{method}: missing rays')
                else:
                    failures += 1
                result.update(case_id=case, repeat=repeat)
                if args.output:
                    for key, arrays in private.items():
                        if isinstance(arrays, dict):
                            name = f'{case}_{method}_r{repeat}{key}.npz'
                            np.savez_compressed(args.output / name, **arrays)
                    with (args.output / 'results.jsonl').open('a') as handle:
                        handle.write(json.dumps(result, default=json_value, allow_nan=False) + '\n')
                print(f'{case} r={repeat} {method}: {result["state"]}, '
                      f'{result["wall_total"]:.4f}s, LP={result["lp_calls"]}' +
                      (f', {result.get("failure_reason", "")}' if result['state'] != 'success' else ''),
                      flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
