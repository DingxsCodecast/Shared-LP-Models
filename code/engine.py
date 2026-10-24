"""Signed-extension experiment engine, public HiGHS API, IEEE interval checks.

The timed four-method implementation currently requires a certified positive
origin margin. Zero-margin/boundary identities are exercised separately by E0.
Stored float64 instance data are interpreted as exact binary rational values.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import resource
import time
from fractions import Fraction

import highspy
import numpy as np
from scipy import sparse

EPS = np.finfo(np.float64).eps
ETA = np.nextafter(0., 1.)
INF = highspy.kHighsInf
TARGET = 1e-7


def down(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


def exact_endpoint_sum(values, upward):
    """Sum binary64 endpoints as integers, then round once outwards."""
    pairs=[float(v).as_integer_ratio() for v in values if v!=0.]
    if not pairs: return 0.
    power=max(den.bit_length()-1 for _,den in pairs)
    total=sum(num << (power-(den.bit_length()-1)) for num,den in pairs)
    rounded=float(Fraction(total,1 << power))
    return float(up(rounded) if upward else down(rounded))


def dot_iv(a, x):
    """Enclose row dot products; multiplication endpoints then sum error bound.

    For n additions, gamma_(n-1) <= 2*n*eps when n*eps <= 1/4.
    The absolute sum is itself bounded upwards. 2*n*ETA covers underflow.
    Reordered/pairwise reductions use no more than n-1 additions.
    All operations are float64, finite, round-to-nearest, no fast-math.
    """
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    p = a*x
    if not np.all(np.isfinite(p)):
        raise ArithmeticError('nonfinite interval product')
    n = p.shape[-1]
    if n*EPS > .25:
        raise ArithmeticError('unsupported interval reduction length')
    lo, hi = down(p), up(p)
    if p.ndim==1:
        # Tight scalar sums avoid a dimension-dependent certificate floor.
        zero=(a==0.) | (x==0.)
        lo=np.where(zero,0.,lo);hi=np.where(zero,0.,hi)
        return exact_endpoint_sum(lo,False),exact_endpoint_sum(hi,True)
    gamma = up(2*n*EPS)
    tiny = up(2*n*ETA)
    ab = np.maximum(np.sum(np.abs(lo), axis=-1), np.sum(np.abs(hi), axis=-1))
    ab_up = up(up(ab+tiny)/down(1-gamma))
    err = up(up(gamma*ab_up)+tiny)
    return down(np.sum(lo, axis=-1)-err), up(np.sum(hi, axis=-1)+err)


def padded_rows(matrix):
    matrix = matrix.tocsr()
    width = int(np.max(np.diff(matrix.indptr), initial=0))
    values = np.zeros((matrix.shape[0], max(1, width)))
    indices = np.zeros_like(values, dtype=np.int32)
    for row in range(matrix.shape[0]):
        s, e = matrix.indptr[row:row+2]
        values[row, :e-s] = matrix.data[s:e]
        indices[row, :e-s] = matrix.indices[s:e]
    return values, indices


def sparse_iv(padded, x):
    values, indices = padded
    return dot_iv(values, np.asarray(x)[indices])


class Stats:
    def __init__(self):
        self.times = dict(model_build=0., lp_solve=0., certificate=0.,certificate_lp=0.,
                          pool_scan=0., pool_verify=0., qy_compute=0.)
        self.lp_calls = 0
        self.pivots = 0
        self.warm_calls = 0
        self.zero_pivot_warm_calls = 0
        self.pool_peak = 0
        self.pool_improved = 0
        self.oracle_calls = 0
        self.classification_calls = 0
        self.origin_calls = 0
        self.direct_calls = 0
        self.polish_calls = 0
        self.fallback_calls = 0
        self.model_warnings = 0
        self.dropped_coefficients = 0


class LP:
    def __init__(self, matrix, rhs, cost, stats, row_lower=None, col_lower=None,
                 col_upper=None, presolve='choose', time_limit=30.):
        start = time.perf_counter()
        self.stats = stats
        self.h = highspy.Highs()
        opts = dict(output_flag=False, solver='simplex', threads=1, parallel='off',
                    presolve=presolve, primal_feasibility_tolerance=1e-10,
                    dual_feasibility_tolerance=1e-10, time_limit=time_limit,
                    random_seed=20261002, allow_unbounded_or_infeasible=False)
        for name, value in opts.items():
            if self.h.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f'HiGHS rejected option {name}')
        mat = matrix.tocsc()
        nrow, ncol = mat.shape
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = ncol, nrow
        lp.col_cost_ = np.asarray(cost, dtype=float)
        lp.col_lower_ = np.zeros(ncol) if col_lower is None else col_lower
        lp.col_upper_ = np.full(ncol, INF) if col_upper is None else col_upper
        lp.row_lower_ = np.full(nrow, -INF) if row_lower is None else row_lower
        lp.row_upper_ = rhs
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = mat.indptr.astype(np.int32)
        lp.a_matrix_.index_ = mat.indices.astype(np.int32)
        lp.a_matrix_.value_ = mat.data
        pass_status=self.h.passModel(lp)
        if pass_status == highspy.HighsStatus.kError:
            raise RuntimeError('passModel failed')
        self.stats.model_warnings += int(pass_status == highspy.HighsStatus.kWarning)
        self.stats.dropped_coefficients += max(0,mat.nnz-self.h.getNumNz())
        self.ncol = ncol
        self.indices = np.arange(ncol, dtype=np.int32)
        self.solved = False
        self.stats.times['model_build'] += time.perf_counter()-start

    def solve(self, cost=None, role='oracle'):
        start = time.perf_counter()
        if cost is not None:
            if self.h.changeColsCost(self.ncol, self.indices, np.asarray(cost)) != highspy.HighsStatus.kOk:
                raise RuntimeError('changeColsCost failed')
        warm = self.solved and self.h.getBasis().valid
        status = self.h.run()
        model_status = self.h.getModelStatus()
        info = self.h.getInfo()
        self.stats.lp_calls += 1
        self.stats.pivots += max(0, info.simplex_iteration_count)
        self.stats.warm_calls += int(warm)
        self.stats.zero_pivot_warm_calls += int(warm and info.simplex_iteration_count == 0)
        if role == 'oracle': self.stats.oracle_calls += 1
        elif role == 'classify': self.stats.classification_calls += 1
        elif role == 'origin': self.stats.origin_calls += 1
        elif role == 'certificate': self.stats.polish_calls += 1
        else: self.stats.direct_calls += 1
        self.stats.times['certificate_lp' if role=='certificate' else 'lp_solve'] += time.perf_counter()-start
        self.solved = model_status == highspy.HighsModelStatus.kOptimal
        if model_status == highspy.HighsModelStatus.kUnbounded:
            return None
        if status == highspy.HighsStatus.kError or model_status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError('LP status: '+self.h.modelStatusToString(model_status))
        sol = self.h.getSolution()
        return dict(x=np.array(sol.col_value), dual=np.array(sol.row_dual),
                    value=float(info.objective_function_value))


@dataclass
class Instance:
    G: object
    h: np.ndarray
    upper: np.ndarray
    a: np.ndarray
    a0: float
    B: np.ndarray
    b0: np.ndarray
    interior: np.ndarray
    meta: dict


class Context:
    def __init__(self, instance, stats):
        self.d = instance
        self.s = stats
        start = time.perf_counter()
        self.gt = padded_rows(instance.G.T)
        self.gr = padded_rows(instance.G)
        self.absG = abs(instance.G)
        self.best_lower=np.full(len(instance.b0),-np.inf)
        self.lower_witness=np.zeros((len(instance.b0),len(instance.a)))
        self.last_certificate=None
        self.s.times['model_build'] += time.perf_counter()-start

    def bound(self, p, constant, dual, q=None, b=None, b0=None):
        """Rigorous min-objective lower bound using a sign-valid dual vector.

        When q is provided, enclose exact b+q*a and b0+q*a0 rather than
        silently interpreting the rounded LP cost as the desired cost.
        """
        start = time.perf_counter()
        v = np.minimum(np.asarray(dual), 0.)
        _, gt_hi = sparse_iv(self.gt, v)
        if q is None:
            p_lo = np.asarray(p)
            const_lo = float(constant)
        else:
            p_lo = down(np.asarray(b)+down(q*self.d.a))
            const_lo = float(down(b0+down(q*self.d.a0)))
        e_hi = up(gt_hi-p_lo)
        bad = np.maximum(e_hi, 0.)
        _, correction = dot_iv(bad, self.d.upper)
        hv_lo, _ = dot_iv(v, self.d.h)
        result = float(down(down(const_lo+hv_lo)-correction))
        if not math.isfinite(result): raise ArithmeticError('nonfinite certificate')
        self.s.times['certificate'] += time.perf_counter()-start
        return result

    def feasible(self, raw):
        start = time.perf_counter()
        d = self.d
        if d.meta['family'] in ('transportation', 'simplex'):
            y = np.rint(raw)
            # These families use bounded integer data; integer sums are exact.
            yi = y.astype(np.int64)
            vals = d.G.astype(np.int64) @ yi
            if np.any(yi < 0) or np.any(yi > d.upper) or np.any(vals > d.h):
                raise RuntimeError('integer primal verification failed')
        else:
            raw = np.clip(raw, 0., d.upper)
            theta = 0.
            for attempt in range(45):
                y = raw if theta == 0 else d.interior+(1-theta)*(raw-d.interior)
                _, hi = sparse_iv(self.gr, y)
                if np.all(y >= 0) and np.all(y <= d.upper) and np.all(hi <= d.h): break
                theta = 1e-12 if theta == 0 else theta*2
            else:
                raise RuntimeError('primal interval repair failed')
        self.s.times['pool_verify'] += time.perf_counter()-start
        return y

    def slope_bound(self, j, solution, point):
        lower=self.bound(self.d.B[j],self.d.b0[j],solution['dual'])
        # Transportation/simplex constraint matrices are totally unimodular,
        # with integral rhs/bounds. For integer costs the optimum is integer.
        # A verified primal value matching ceil(dual lower bound) proves it.
        b=self.d.B[j]
        if self.d.meta['family'] in ('transportation','simplex') and np.all(b==np.rint(b)):
            l0=self.bound(b,0.,solution['dual'])
            exact_primal=int(b.astype(np.int64)@point['y'].astype(np.int64))
            if math.ceil(l0)==exact_primal:
                lower=max(lower,float(down(self.d.b0[j]+exact_primal)))
        return lower

    def point(self, raw, j=None):
        raw=np.asarray(raw)
        # Nominal ratios steer Newton only. They are NEVER used as certified
        # lower bounds; feasibility-repaired interval ratios below supply those.
        raw_A=float(self.d.a0+self.d.a@raw)
        qnom=np.full(len(self.d.b0),np.nan)
        if raw_A>0:
            if j is None: qnom=-(self.d.b0+self.d.B@raw)/raw_A
            else: qnom[j]=-(self.d.b0[j]+self.d.B[j]@raw)/raw_A
        y = self.feasible(raw)
        start = time.perf_counter()
        alo, ahi = dot_iv(self.d.a, y)
        alo = float(down(alo+self.d.a0)); ahi = float(up(ahi+self.d.a0))
        if j is None:
            blo, bhi = dot_iv(self.d.B, y)
            blo, bhi = down(blo+self.d.b0), up(bhi+self.d.b0)
        else:
            bl,bh=dot_iv(self.d.B[j],y)
            blo=np.full(len(self.d.b0),np.nan); bhi=blo.copy()
            blo[j]=down(bl+self.d.b0[j]); bhi[j]=up(bh+self.d.b0[j])
        if alo <= 0: raise RuntimeError('positive-origin-margin implementation requires A>0')
        numerator = -bhi
        qlo = down(np.minimum(numerator/alo, numerator/ahi))
        improve=qlo>self.best_lower
        self.best_lower[improve]=qlo[improve]
        self.lower_witness[improve]=y
        self.s.times['qy_compute'] += time.perf_counter()-start
        return dict(y=y, alo=alo, ahi=ahi, blo=blo, bhi=bhi, qlo=qlo,qnom=qnom)

    def coefficient(self, q, dual, j, delta):
        L = self.bound(None, None, dual, q=q, b=self.d.B[j], b0=self.d.b0[j])
        start = time.perf_counter()
        correction = float(up(up(max(0., -L))/delta))
        safe = float(up(q+correction))
        # Monotonicity: F(q+correction) >= L+correction*delta >= 0.
        for attempt in range(12):
            effective_lo=float(down(safe-q))
            if float(down(L+down(effective_lo*delta)))>=0: break
            safe=float(up(safe+max(ETA,abs(safe)*8*EPS,abs(correction)*8*EPS)))
        else: raise ArithmeticError('coefficient rounding verification unavailable')
        if safe < q: raise ArithmeticError('unsafe coefficient rounding')
        correction=float(up(safe-q))
        self.last_certificate=dict(q_nom=q,dual=np.minimum(np.asarray(dual),0.).copy(),
                                   q_safe=safe,lower_before=L,delta=delta)
        self.s.times['certificate'] += time.perf_counter()-start
        return safe, L, correction


def positive_lp(d, j, stats, role='positive_direct'):
    b = d.B[j]
    mat = sparse.vstack([
        sparse.hstack([sparse.csr_matrix((-b).reshape(-1,1)), d.G.T]),
        sparse.csr_matrix(np.r_[-d.b0[j], -d.h].reshape(1,-1))], format='csc')
    rows = d.G.shape[0]
    lp = LP(mat, np.r_[d.a, d.a0], np.r_[-1., np.zeros(rows)], stats,
            col_lower=np.r_[0., np.full(rows,-INF)],
            col_upper=np.r_[INF, np.zeros(rows)])
    return lp.solve(role=role)


def negative_lp(d, j, stats, role='negative_direct'):
    mat = sparse.vstack([
        sparse.hstack([d.G, sparse.csr_matrix((-d.h).reshape(-1,1))]),
        sparse.csr_matrix(np.r_[d.B[j],d.b0[j]].reshape(1,-1))],format='csc')
    rhs = np.r_[np.zeros(d.G.shape[0]),1.]
    row_lo = np.r_[np.full(d.G.shape[0],-INF),1.]
    return LP(mat,rhs,-np.r_[d.a,d.a0],stats,row_lower=row_lo).solve(role=role)


def run_method(d, method, target=TARGET, pool_max=256):
    start = time.perf_counter()
    s = Stats(); ctx = Context(d,s)
    inner = LP(d.G,d.h,d.a,s)
    result = dict(method=method, state='success', rays=[], meta=d.meta)
    proofs=[];origin_proof=None
    pool = []

    def add(point):
        if method != 'NP': return
        t = time.perf_counter()
        # Deterministic exact stored-vector deduplication and FIFO eviction.
        if not any(np.array_equal(point['y'],p['y']) for p in pool):
            pool.append(point)
            if len(pool)>pool_max: pool.pop(0)
        s.pool_peak = max(s.pool_peak,len(pool))
        s.times['pool_scan'] += time.perf_counter()-t

    def pool_lower(j, candidate):
        t = time.perf_counter()
        value = max([candidate]+[float(p['qlo'][j]) for p in pool])
        s.pool_improved += int(value > candidate+1e-14)
        s.times['pool_scan'] += time.perf_counter()-t
        return value

    try:
        origin = inner.solve(d.a, role='origin')
        delta = ctx.bound(d.a,d.a0,origin['dual'])
        origin_proof=np.minimum(origin['dual'],0.).copy()
        if delta <= 0:
            raise RuntimeError('origin positive margin unavailable; zero-margin uses E0 reference')
        if method=='NP': add(ctx.point(origin['x']))
        k = len(d.b0)
        branches = []; initial=[]; direct=[]
        # All method-specific classification/preprocessing is timed.
        for j in range(k):
            if time.perf_counter()-start > 300: raise TimeoutError('batch time limit')
            if method in ('DL','D24'):
                pos = positive_lp(d,j,s); direct.append(pos)
                if pos is not None:
                    lam = pos['x'][0]
                    if lam <= 0: raise RuntimeError('zero-positive-distance')
                    t = -pos['dual'][-1]
                    if t <= 0: raise RuntimeError('positive dual point recovery failed')
                    point = ctx.point(-pos['dual'][:-1]/t,j)
                    branches.append('positive'); initial.append(point)
                    continue
            else:
                direct.append(None)
                witness = None
                if method == 'NP':
                    for p in pool:
                        if p['bhi'][j] < 0: witness=p; break
                if witness is not None:
                    branches.append('positive'); initial.append(witness)
                    continue
            slope = inner.solve(d.B[j],role='classify')
            point = ctx.point(slope['x'],None if method=='NP' else j); add(point)
            beta_L = ctx.slope_bound(j,slope,point)
            if point['bhi'][j]<0:
                branches.append('positive'); initial.append(point)
            elif beta_L>0:
                branches.append('negative'); initial.append(point)
            else:
                raise RuntimeError('classification-unresolved')
        if all(b=='negative' for b in branches):
            result['state']='no-improvement'
        else:
            for j,branch in enumerate(branches):
                if time.perf_counter()-start>300: raise TimeoutError('batch time limit')
                qlower=float(initial[j]['qlo'][j])
                qs=None; raw_distance=None; L=None; correction=None
                def refine_certificate(q,qlower,qs,L,correction,force=False):
                    # The candidate is already near a verified lower bracket,
                    # but a scaled Newton/direct-LP dual may be too inaccurate.
                    # All four methods use this SAME conditional, paid-for LP.
                    qnext=None
                    if (qs-qlower>target*(1+abs(qs)) and
                            (force or abs(q-qlower)<=target*(1+abs(q)))):
                        cert=inner.solve(d.B[j]+q*d.a,role='certificate')
                        p=ctx.point(cert['x'],None if method=='NP' else j);add(p)
                        qlower=max(qlower,float(p['qlo'][j]))
                        qs,L,correction=ctx.coefficient(q,cert['dual'],j,delta)
                        qnext=max(qlower,float(p['qnom'][j]))
                        if force and qs-qlower>target*(1+abs(qs)):
                            # Common safeguard against a stalled, inaccurate
                            # retained basis. Rebuild the endpoint reformulation;
                            # both its model and solve are charged to this run.
                            s.fallback_calls+=1
                            if branch=='positive':
                                fresh=positive_lp(d,j,s,role='certificate')
                                if fresh is None or fresh['x'][0]<=0:
                                    raise RuntimeError('positive fallback unresolved')
                                lam=float(fresh['x'][0]);qfresh=1/lam
                                t=-fresh['dual'][-1]
                                if t<=0:raise RuntimeError('fallback witness unavailable')
                                p=ctx.point(-fresh['dual'][:-1]/t,None if method=='NP' else j)
                                v=fresh['x'][1:]/lam
                            else:
                                fresh=negative_lp(d,j,s,role='certificate')
                                if fresh is None or fresh['x'][-1]<=0:
                                    raise RuntimeError('negative fallback unresolved')
                                gamma=float(-fresh['dual'][-1])
                                if gamma<=0:raise RuntimeError('fallback gamma unavailable')
                                qfresh=-1/gamma;v=fresh['dual'][:-1]/gamma
                                p=ctx.point(fresh['x'][:-1]/fresh['x'][-1],None if method=='NP' else j)
                            add(p);qlower=max(qlower,float(p['qlo'][j]))
                            qs,L,correction=ctx.coefficient(qfresh,v,j,delta)
                            qnext=max(qlower,float(p['qnom'][j]))
                            if qs-qlower>target*(1+abs(qs)):
                                raise RuntimeError('coefficient-quality-unresolved after uniform fallback')
                    return qlower,qs,L,correction,qnext
                if branch=='positive' and method in ('DL','D24'):
                    sol=direct[j]; lam=float(sol['x'][0]); q=1/lam
                    qs,L,correction=ctx.coefficient(q,sol['x'][1:]/lam,j,delta)
                    qlower,qs,L,correction,_=refine_certificate(q,qlower,qs,L,correction,True)
                    raw_distance=lam
                elif branch=='negative' and method=='DL':
                    sol=negative_lp(d,j,s)
                    if sol is None: raise RuntimeError('negative CC unbounded after positive beta')
                    if sol['x'][-1]<=0: raise RuntimeError('negative CC t nonpositive')
                    point=ctx.point(sol['x'][:-1]/sol['x'][-1],j)
                    qlower=max(qlower,float(point['qlo'][j]))
                    gamma=float(-sol['dual'][-1])
                    if gamma<=0: raise RuntimeError('negative dual gamma nonpositive')
                    q=-1/gamma
                    qs,L,correction=ctx.coefficient(q,sol['dual'][:-1]/gamma,j,delta)
                    qlower,qs,L,correction,_=refine_certificate(q,qlower,qs,L,correction,True)
                    raw_distance=-sol['value']
                else:
                    if method=='NP': qlower=pool_lower(j,qlower)
                    qcandidate=max(qlower,float(initial[j]['qnom'][j]))
                    for it in range(200):
                        if time.perf_counter()-start>300: raise TimeoutError('batch time limit')
                        if branch=='positive':
                            if qcandidate<=0: raise RuntimeError('positive initialization unavailable')
                            distance=float(1/qcandidate); q=1/distance
                            sol=inner.solve(d.a+distance*d.B[j])
                        else:
                            if qcandidate>=0: raise RuntimeError('negative initialization unavailable')
                            distance=float(-1/qcandidate); q=-1/distance
                            sol=inner.solve(distance*d.B[j]-d.a)
                        point=ctx.point(sol['x'],None if method=='NP' else j); add(point)
                        qlower=max(qlower,float(point['qlo'][j]))
                        if method=='NP': qlower=pool_lower(j,qlower)
                        qs,L,correction=ctx.coefficient(q,sol['dual']/distance,j,delta)
                        next_nom=max(qlower,float(point['qnom'][j]))
                        stalled=abs(next_nom-q)<=1e-12*(1+abs(q))
                        qlower,qs,L,correction,polished_nom=refine_certificate(
                            q,qlower,qs,L,correction,stalled)
                        raw_distance=distance
                        if qs-qlower<=target*(1+abs(qs)): break
                        qcandidate=next_nom if polished_nom is None else polished_nom
                    else: raise RuntimeError('Newton precision iteration limit')
                if qs is None or qs<qlower-1e-13 or qs-qlower>target*(1+abs(qs)):
                    raise RuntimeError('coefficient-quality-unresolved')
                raw_distance=abs(1/ctx.last_certificate['q_nom'])
                result['rays'].append(dict(j=j,branch=branch,q_safe=qs,q_lower=qlower,
                                           q_width=qs-qlower,raw_distance=raw_distance,
                                           certificate_lower_before_correction=L,
                                           coefficient_correction=correction,
                                           certificate_status='interval_verified'))
                proofs.append(dict(**ctx.last_certificate,q_lower=qlower,
                                   lower_witness=ctx.lower_witness[j].copy(),j=j))
    except Exception as exc:
        result['state']='timeout' if isinstance(exc,TimeoutError) or 'Time limit' in str(exc) else 'failed'
        result['failure_reason']=str(exc)
    result['wall_total']=time.perf_counter()-start
    if origin_proof is not None:
        result['_proof']=dict(origin_dual=origin_proof,origin_delta_lower=delta,
                              ray_indices=np.array([p['j'] for p in proofs],dtype=int),
                              cert_q_nom=np.array([p['q_nom'] for p in proofs]),
                              cert_dual=np.array([p['dual'] for p in proofs]),
                              q_safe=np.array([p['q_safe'] for p in proofs]),
                              q_lower=np.array([p['q_lower'] for p in proofs]),
                              lower_before=np.array([p['lower_before'] for p in proofs]),
                              lower_witness=np.array([p['lower_witness'] for p in proofs]))
    result.update(s.times)
    result.update(lp_calls=s.lp_calls,simplex_iterations=s.pivots,warm_calls=s.warm_calls,
                  zero_pivot_warm_calls=s.zero_pivot_warm_calls,pool_size_peak=s.pool_peak,
                  pool_bound_improved=s.pool_improved,oracle_calls=s.oracle_calls,
                  classification_calls=s.classification_calls,origin_calls=s.origin_calls,
                  direct_calls=s.direct_calls,model_warnings=s.model_warnings,
                  polish_calls=s.polish_calls,
                  fallback_calls=s.fallback_calls,
                  dropped_coefficients=s.dropped_coefficients,
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    return result


def generate(family,m,k,positive_fraction,seed,correlation=0.,beta_margin=1.):
    rng=np.random.default_rng(seed)
    if family=='bounded_sparse_coupled':
        H=sparse.random(4*m,m,density=min(.1,16/m),random_state=rng,format='csr',
                        data_rvs=lambda n:rng.choice([-3,-2,-1,1,2,3],size=n))
        interior=np.full(m,.5)
        hH=H@interior+rng.choice([.5,1.,2.],size=4*m)
        G=sparse.vstack([H,sparse.eye(m)],format='csr')
        h=np.r_[hH,np.ones(m)]; upper=np.ones(m)
    elif family=='transportation':
        p=8 if m in (64,128) else 16
        q=m//p
        flow=rng.integers(1,4,size=(p,q))
        supply=flow.sum(axis=1); demand=flow.sum(axis=0)
        eq=sparse.vstack([sparse.kron(sparse.eye(p),np.ones((1,q))),
                          sparse.kron(np.ones((1,p)),sparse.eye(q))],format='csr')
        rhs=np.r_[supply,demand]
        upper=np.minimum(supply[:,None],demand[None,:]).ravel().astype(float)
        G=sparse.vstack([eq,-eq,sparse.eye(m)],format='csr')
        h=np.r_[rhs,-rhs,upper].astype(float); interior=flow.ravel().astype(float)
    else: raise ValueError(family)
    a=rng.integers(-10,11,size=m).astype(float)
    noise=rng.integers(-5,6,size=(k,m)).astype(float)
    common=rng.integers(-5,6,size=m).astype(float)
    # Integer data are used for the base grid; correlation ablations are dyadic.
    B=correlation*common+(1-correlation)*noise
    b0=np.zeros(k)
    meta=dict(family=family,m=m,k=k,positive_fraction=positive_fraction,seed=seed,
              correlation=correlation,beta_margin=beta_margin)
    d=Instance(G,h,upper,a,0.,B,b0,interior,meta)
    stats=Stats(); ctx=Context(d,stats); lp=LP(G,h,a,stats)
    start=time.perf_counter()
    origin=lp.solve(a,role='origin')
    lower=ctx.bound(a,0.,origin['dual'])
    d.a0=float(up(-lower+1.))
    positive=np.zeros(k,dtype=bool); positive[:round(k*positive_fraction)]=True
    rng.shuffle(positive)
    for j in range(k):
        sol=lp.solve(B[j],role='classify')
        lower=ctx.bound(B[j],0.,sol['dual'])
        y=ctx.feasible(sol['x'])
        _,upper_value=dot_iv(B[j],y)
        if family=='transportation' and np.all(B[j]==np.rint(B[j])):
            value=int(B[j].astype(np.int64)@y.astype(np.int64))
            if math.ceil(lower)==value:
                lower=float(value);upper_value=float(value)
        positive_margin=max(1.,beta_margin)
        if positive[j]: b0[j]=float(down(-float(upper_value)-positive_margin))
        else: b0[j]=float(up(-lower+beta_margin))
    meta.update(rows=G.shape[0],nnz=G.nnz,generation_s=time.perf_counter()-start,
                expected_branches=['positive' if p else 'negative' for p in positive],
                data_interpretation='exact stored binary64 values',
                positive_slope_margin=max(1.,beta_margin))
    return d


def save_instance(d,path):
    path=Path(path)
    np.savez_compressed(path,data=d.G.data,indices=d.G.indices,indptr=d.G.indptr,
                        shape=d.G.shape,h=d.h,upper=d.upper,a=d.a,a0=d.a0,B=d.B,
                        b0=d.b0,interior=d.interior,meta=json.dumps(d.meta))


def load_instance(path):
    with np.load(path,allow_pickle=False) as z:
        G=sparse.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
        return Instance(G,z['h'],z['upper'],z['a'],float(z['a0']),z['B'],z['b0'],
                        z['interior'],json.loads(str(z['meta'])))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
