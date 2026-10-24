"""Follow-up controller: preserve negative signs with paid slope-margin certificates."""
from fractions import Fraction
import resource
import time
import numpy as np
from scipy import sparse
import highspy
import engine


def secant_upper(left, right):
    """Concave increasing F: outward root of the chord of reliable lower values.

    left=(evaluated q, F lower, F upper), F upper<0; right lower>0.
    The chord is below F, so its root is an upper bound for q*. None of the
    computed roots is reused as an evaluated function anchor.
    """
    l, fl, fu = left
    u, gu, _ = right
    if not (l < u and fu < 0 and fl <= fu and gu > 0): return None
    l, u, fl, gu = map(Fraction, (float(l), float(u), float(fl), float(gu)))
    exact = l + (-fl)*(u-l)/(gu-fl)
    value = float(exact)
    if Fraction(value) < exact: value = float(engine.up(value))
    assert Fraction(value) >= exact
    return value


def run_structure(d, method, target=engine.TARGET, pool_max=256):
    start=time.perf_counter(); stats=engine.Stats(); ctx=engine.Context(d,stats)
    inner=engine.LP(d.G,d.h,d.a,stats)
    use_pool=method in ('SP','SG'); guard=method=='SG'
    result=dict(method=method,state='success',meta=d.meta,rays=[])
    pool=[];keys=set();saved={};proofs=[];origin_dual=None; slope_certificates={}; A_upper=None
    counts=dict(basis_restore_calls=0,first_oracle_hits=0,guard_anchor_calls=0,
                midpoint_calls=0,secant_bounds=0,positive_first_hits=0,
                negative_first_hits=0,point_pool_bytes_peak=0,margin_bound_calls=0,margin_bound_time_s=0.)
    interval_time=0.;cache_time=0.

    def check_time():
        if time.perf_counter()-start>300: raise TimeoutError('batch time limit')

    def point(raw,j):
        p=ctx.point(raw,None if use_pool else j)
        if use_pool:
            t=time.perf_counter();key=p['y'].tobytes()
            if key not in keys:
                keys.add(key);pool.append(p)
                if len(pool)>pool_max:keys.remove(pool.pop(0)['y'].tobytes())
            stats.pool_peak=max(stats.pool_peak,len(pool))
            counts['point_pool_bytes_peak']=max(counts['point_pool_bytes_peak'],sum(
                a.nbytes for p in pool for a in p.values() if isinstance(a,np.ndarray)))
            stats.times['pool_scan']+=time.perf_counter()-t
        return p

    def pool_candidate(j,initial):
        t=time.perf_counter()
        lower=float(ctx.best_lower[j])
        candidates=[float(initial['qnom'][j])]
        if use_pool:candidates += [float(p['qnom'][j]) for p in pool]
        candidates=[x for x in candidates if np.isfinite(x)]
        candidate=max([lower]+candidates)
        stats.times['pool_scan']+=time.perf_counter()-t
        return lower,candidate

    def common_fallback(j):
        stats.fallback_calls+=1
        mat=sparse.vstack([sparse.hstack([d.G,sparse.csr_matrix((-d.h)[:,None])]),
            sparse.csr_matrix(np.r_[d.a,d.a0][None,:])],format='csc')
        fresh=engine.LP(mat,np.r_[np.zeros(len(d.h)),1.],np.r_[d.B[j],d.b0[j]],stats,
            row_lower=np.r_[np.full(len(d.h),-engine.INF),1.])
        sol=fresh.solve(role='certificate')
        if sol is None or sol['x'][-1]<=0:raise RuntimeError('common fallback unresolved')
        p=point(sol['x'][:-1]/sol['x'][-1],j)
        q=-float(sol['dual'][-1])
        safe,L,corr=ctx.coefficient(q,sol['dual'][:-1],j,delta)
        return p,safe,L,corr

    try:
        origin=inner.solve(d.a,role='origin')
        delta=ctx.bound(d.a,d.a0,origin['dual'])
        origin_dual=np.minimum(origin['dual'],0.)
        if delta<=0:raise RuntimeError('positive origin margin unavailable')
        if use_pool:point(origin['x'],None)
        branches=[];initial=[]
        for j in range(len(d.b0)):
            check_time();t=time.perf_counter()
            witnesses=[p for p in pool if p['bhi'][j]<0] if use_pool else []
            witness=max(witnesses,key=lambda p:p['qlo'][j]) if witnesses else None
            stats.times['pool_scan']+=time.perf_counter()-t
            if witness is not None:
                branches.append('positive');initial.append(witness);continue
            inner.h.setOptionValue('simplex_strategy',1)
            slope=inner.solve(d.B[j],role='classify');p=point(slope['x'],j)
            beta=ctx.slope_bound(j,slope,p)
            slope_certificates[j]=(beta,np.minimum(slope['dual'],0.).copy())
            t=time.perf_counter();basis=inner.h.getBasis()
            if not basis.valid:raise RuntimeError('classification basis unavailable')
            saved[j]=basis;cache_time+=time.perf_counter()-t
            if p['bhi'][j]<0:branch='positive'
            elif beta>0:branch='negative'
            else:raise RuntimeError('classical branch classification unresolved')
            branches.append(branch);initial.append(p)
        if all(b=='negative' for b in branches):
            result['state']='no-improvement'
        else:
            inner.h.setOptionValue('simplex_strategy',4)
            for j,branch in enumerate(branches):
                check_time();t=time.perf_counter()
                if j in saved:
                    if inner.h.setBasis(saved[j])!=highspy.HighsStatus.kOk:
                        raise RuntimeError('stored basis not accepted')
                    counts['basis_restore_calls']+=1
                cache_time+=time.perf_counter()-t
                lower,q=pool_candidate(j,initial[j])
                best_safe=float('inf');best_cert=None;left=right=None
                stagnant=bad_shrink=0;previous_width=float('inf');need_anchor=False
                pending_anchor=False;oracle_count=0;midpoint_count=0
                ray_start=time.perf_counter();p=initial[j]
                for iteration in range(200):
                    check_time()
                    sol=inner.solve(d.B[j]+q*d.a)
                    p=point(sol['x'],j);oracle_count+=1
                    lower,next_q=pool_candidate(j,p)
                    safe,L,correction=ctx.coefficient(q,sol['dual'],j,delta)
                    if safe<best_safe:
                        best_safe=safe;best_cert=dict(ctx.last_certificate)
                    if lower>best_safe:raise RuntimeError('inconsistent coefficient bounds')
                    width=best_safe-lower
                    if branch=='negative' and best_safe>=0:
                        mt=time.perf_counter()
                        beta,sv=slope_certificates[j]
                        if A_upper is None:
                            exact_A=Fraction(float(d.a0))+sum((max(Fraction(float(a)),0)*Fraction(float(u))
                                for a,u in zip(d.a,d.upper)),Fraction(0))
                            A_upper=float(exact_A)
                            if Fraction(A_upper)<exact_A:A_upper=float(engine.up(A_upper))
                            if not np.isfinite(A_upper) or A_upper<=0:raise RuntimeError('A upper bound unavailable')
                        exact_q=-Fraction(float(beta))/Fraction(A_upper)
                        negative_upper=float(exact_q)
                        if Fraction(negative_upper)<exact_q:negative_upper=float(engine.up(negative_upper))
                        counts['margin_bound_calls']+=1
                        if lower<=negative_upper<0 and negative_upper-lower<=target*(1+abs(negative_upper)):
                            best_safe=negative_upper
                            best_cert=dict(q_nom=negative_upper,dual=sv,q_safe=negative_upper,
                                lower_before=beta,certificate_kind=1,slope_beta=beta,A_upper=A_upper)
                            width=best_safe-lower
                        counts['margin_bound_time_s']+=time.perf_counter()-mt
                    if width<=target*(1+abs(best_safe)) and (lower>0 if branch=='positive' else best_safe<0):break
                    stagnant=stagnant+1 if abs(next_q-q)<=1e-12*(1+abs(q)) else 0
                    if stagnant>=2:
                        p,fs,_,_=common_fallback(j)
                        lower=float(ctx.best_lower[j])
                        if fs<best_safe:best_safe=fs;best_cert=dict(ctx.last_certificate)
                        break
                    if not guard:q=next_q;continue
                    t=time.perf_counter()
                    # A feasible point's enclosed affine value is an upper F bound.
                    aend=p['ahi'] if q>=0 else p['alo']
                    f_upper=float(engine.up(p['bhi'][j]+engine.up(q*aend)))
                    if f_upper<0:
                        if left is None or q>left[0]:left=(q,L,f_upper)
                    elif L>0:
                        if right is None or q<right[0]:right=(q,L,f_upper)
                    search_upper=best_safe
                    if left is not None and right is not None:
                        upper=secant_upper(left,right)
                        if upper is not None:
                            search_upper=min(search_upper,upper);counts['secant_bounds']+=1
                    search_width=search_upper-lower
                    if search_width<0:raise RuntimeError('invalid safeguarded interval')
                    bad_shrink=bad_shrink+1 if search_width>0.5*previous_width else 0
                    previous_width=search_width
                    if left is not None and right is None and not pending_anchor:
                        # This extra evaluation is charged. The interval itself
                        # is not expanded merely to create an evaluated anchor.
                        q=float(engine.up(best_safe+target*(1+abs(best_safe))))
                        pending_anchor=True;counts['guard_anchor_calls']+=1
                    elif bad_shrink>=2:
                        q=lower+(search_upper-lower)/2
                        bad_shrink=0;counts['midpoint_calls']+=1;midpoint_count+=1
                    else:q=min(max(next_q,lower),search_upper)
                    interval_time+=time.perf_counter()-t
                else:raise RuntimeError('Newton iteration budget exhausted')
                if best_cert is None or not 0<=best_safe-lower<=target*(1+abs(best_safe)):
                    raise RuntimeError('coefficient quality unresolved after fallback')
                if (branch=='positive' and lower<=0) or (branch=='negative' and best_safe>=0):
                    raise RuntimeError('classical coefficient sign unresolved')
                ctx.last_certificate=best_cert
                first_hit=oracle_count==1
                counts['first_oracle_hits']+=int(first_hit)
                counts[branch+'_first_hits']+=int(first_hit)
                result['rays'].append(dict(j=j,branch=branch,q_safe=best_safe,q_lower=lower,
                    q_width=best_safe-lower,raw_distance=abs(1/best_cert['q_nom']),
                    certificate_status='interval_verified',certificate_kind=best_cert.get('certificate_kind',0),oracle_calls=oracle_count,
                    midpoint_calls=midpoint_count,first_oracle_hit=first_hit,
                    ray_time=time.perf_counter()-ray_start))
                proofs.append(dict(**best_cert,q_lower=lower,j=j,
                    lower_witness=ctx.lower_witness[j].copy()))
    except Exception as exc:
        result['state']='timeout' if isinstance(exc,TimeoutError) or 'Time limit' in str(exc) else 'failed'
        result['failure_reason']=str(exc)
    result['wall_total']=time.perf_counter()-start
    if origin_dual is not None:
        result['_proof']=dict(origin_dual=origin_dual,origin_delta_lower=delta,
            ray_indices=np.array([p['j'] for p in proofs],dtype=int),
            cert_q_nom=np.array([p['q_nom'] for p in proofs]),
            cert_dual=np.array([p['dual'] for p in proofs]),
            q_safe=np.array([p['q_safe'] for p in proofs]),
            q_lower=np.array([p['q_lower'] for p in proofs]),
            lower_before=np.array([p['lower_before'] for p in proofs]),
            lower_witness=np.array([p['lower_witness'] for p in proofs]),
            certificate_kind=np.array([p.get('certificate_kind',0) for p in proofs]),
            slope_beta=np.array([p.get('slope_beta',0.) for p in proofs]),
            A_upper=np.array([p.get('A_upper',0.) for p in proofs]))
    result.update(stats.times);result.update(counts)
    result.update(lp_calls=stats.lp_calls,simplex_iterations=stats.pivots,
        oracle_calls=stats.oracle_calls,classification_calls=stats.classification_calls,
        origin_calls=stats.origin_calls,fallback_calls=stats.fallback_calls,
        pool_size_peak=stats.pool_peak,basis_cache_time_s=cache_time,
        interval_maintenance_time_s=interval_time,warm_calls=stats.warm_calls,
        zero_pivot_warm_calls=stats.zero_pivot_warm_calls,
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    return result
