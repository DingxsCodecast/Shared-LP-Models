"""Reviewer control: classification-free normalized Newton/Dinkelbach.

Frozen historical implementations are unchanged. Initialization uses the paid
origin point and the preceding paid oracle point. Only current bases and these
two points are retained. All certification and common-model fallback costs count.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import engine
from strong_baseline import run as run_strong

ROOT = Path(__file__).resolve().parent


def run_nc(d):
    start = time.perf_counter()
    stats = engine.Stats()
    ctx = engine.Context(d, stats)
    inner = engine.LP(d.G, d.h, d.a, stats)
    result = dict(method='NC', state='success', meta=d.meta, rays=[])
    proofs = []
    origin_dual = None
    try:
        origin = inner.solve(d.a, role='origin')
        delta = ctx.bound(d.a, d.a0, origin['dual'])
        origin_dual = np.minimum(origin['dual'], 0.)
        if delta <= 0:
            raise RuntimeError('positive origin margin unavailable')
        initial = ctx.point(origin['x'])
        previous = initial['y']
        inner.h.setOptionValue('simplex_strategy', 4)
        for j in range(len(d.b0)):
            point = ctx.point(previous, j)
            lower = max(float(initial['qlo'][j]), float(point['qlo'][j]))
            q = lower
            stagnant = 0
            for iteration in range(100):
                if time.perf_counter() - start > 300:
                    raise TimeoutError('batch time limit')
                sol = inner.solve(d.B[j] + q*d.a)
                point = ctx.point(sol['x'], j)
                previous = point['y']
                lower = max(lower, float(point['qlo'][j]))
                safe, bound, correction = ctx.coefficient(q, sol['dual'], j, delta)
                if lower <= safe and safe-lower <= engine.TARGET*(1+abs(safe)):
                    break
                next_q = max(lower, float(point['qnom'][j]))
                stagnant = stagnant+1 if abs(next_q-q) <= engine.TARGET*(1+abs(q)) else 0
                q = next_q
                if stagnant >= 2:
                    # Paid fresh common-denominator safeguard, no slope oracle.
                    stats.fallback_calls += 1
                    matrix = sparse.vstack([
                        sparse.hstack([d.G, sparse.csr_matrix((-d.h)[:, None])]),
                        sparse.csr_matrix(np.r_[d.a, d.a0][None, :])], format='csc')
                    fresh = engine.LP(matrix, np.r_[np.zeros(len(d.h)), 1.],
                                      np.r_[d.B[j], d.b0[j]], stats,
                                      row_lower=np.r_[np.full(len(d.h), -engine.INF), 1.])
                    sol = fresh.solve(role='certificate')
                    if sol is None or sol['x'][-1] <= 0:
                        raise RuntimeError('common fallback unresolved')
                    point = ctx.point(sol['x'][:-1]/sol['x'][-1], j)
                    previous = point['y']
                    lower = max(lower, float(point['qlo'][j]))
                    q = -float(sol['dual'][-1])
                    safe, bound, correction = ctx.coefficient(q, sol['dual'][:-1], j, delta)
                    break
            else:
                raise RuntimeError('iteration budget exhausted')
            if not lower <= safe or safe-lower > engine.TARGET*(1+abs(safe)):
                raise RuntimeError('coefficient quality unresolved')
            if lower > 0:
                branch = 'positive'
            elif safe < 0:
                branch = 'negative'
            else:
                raise RuntimeError('classical branch sign unresolved')
            nominal = ctx.last_certificate['q_nom']
            result['rays'].append(dict(j=j, branch=branch, q_safe=safe,
                q_lower=lower, q_width=safe-lower, raw_distance=abs(1/nominal),
                coefficient_correction=correction, certificate_lower_before_correction=bound,
                certificate_status='interval_verified'))
            proofs.append(dict(**ctx.last_certificate, q_lower=lower, j=j,
                               lower_witness=ctx.lower_witness[j].copy()))
    except Exception as exc:
        result['state'] = 'timeout' if isinstance(exc, TimeoutError) or 'Time limit' in str(exc) else 'failed'
        result['failure_reason'] = str(exc)
    result['wall_total'] = time.perf_counter()-start
    if origin_dual is not None:
        result['_proof'] = dict(origin_dual=origin_dual, origin_delta_lower=delta,
            ray_indices=np.array([p['j'] for p in proofs]),
            cert_q_nom=np.array([p['q_nom'] for p in proofs]),
            cert_dual=np.array([p['dual'] for p in proofs]),
            q_safe=np.array([p['q_safe'] for p in proofs]),
            q_lower=np.array([p['q_lower'] for p in proofs]),
            lower_before=np.array([p['lower_before'] for p in proofs]),
            lower_witness=np.array([p['lower_witness'] for p in proofs]))
    result.update(stats.times)
    result.update(lp_calls=stats.lp_calls, simplex_iterations=stats.pivots,
                  classification_calls=stats.classification_calls, origin_calls=stats.origin_calls,
                  fallback_calls=stats.fallback_calls, oracle_calls=stats.oracle_calls,
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    return result
