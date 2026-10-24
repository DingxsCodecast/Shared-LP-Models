"""Exploratory fixed-model A-denominator LP; strict positive origin margin.

This extension leaves the frozen four-method engine untouched. The baseline
is rerun and interleaved with Q on 24 predetermined main inputs (two per cell).
It also compares N and Q on all 20 independently generated within-box inputs.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import sparse

import engine

ROOT=Path(__file__).resolve().parent

def run_q(d):
    start=time.perf_counter();s=engine.Stats();ctx=engine.Context(d,s)
    inner=engine.LP(d.G,d.h,d.a,s)
    r=dict(method='Q',state='success',meta=d.meta,rays=[]);proofs=[];origin_proof=None
    try:
        origin=inner.solve(d.a,role='origin')
        delta=ctx.bound(d.a,d.a0,origin['dual']);origin_proof=np.minimum(origin['dual'],0.)
        if delta<=0:raise RuntimeError('origin positive margin unavailable')
        matrix=sparse.vstack([
            sparse.hstack([d.G,sparse.csr_matrix((-d.h)[:,None])]),
            sparse.csr_matrix(np.r_[d.a,d.a0][None,:])],format='csc')
        cc=engine.LP(matrix,np.r_[np.zeros(len(d.h)),1.],np.r_[d.B[0],d.b0[0]],s,
                     row_lower=np.r_[np.full(len(d.h),-engine.INF),1.])
        for j in range(len(d.b0)):
            if time.perf_counter()-start>300:raise TimeoutError('batch time limit')
            sol=cc.solve(np.r_[d.B[j],d.b0[j]])
            if sol is None or sol['x'][-1]<=0:raise RuntimeError('common CC solve unresolved')
            q=float(-sol['dual'][-1]);p=ctx.point(sol['x'][:-1]/sol['x'][-1],j)
            ql=float(p['qlo'][j]);qs,L,corr=ctx.coefficient(q,sol['dual'][:-1],j,delta)
            # Same paid-for quality safeguard as the frozen four methods.
            if qs-ql>engine.TARGET*(1+abs(qs)):
                cert=inner.solve(d.B[j]+q*d.a,role='certificate')
                p=ctx.point(cert['x'],j);ql=max(ql,float(p['qlo'][j]))
                qs,L,corr=ctx.coefficient(q,cert['dual'],j,delta)
            if ql>0:branch='positive'
            elif qs<0:branch='negative'
            else:
                slope=inner.solve(d.B[j],role='classify');p=ctx.point(slope['x'],j)
                if p['bhi'][j]<0:branch='positive'
                elif ctx.slope_bound(j,slope,p)>0:branch='negative'
                else:raise RuntimeError('classification unresolved around zero')
            if qs-ql>engine.TARGET*(1+abs(qs)):
                s.fallback_calls+=1
                if branch=='positive':
                    fresh=engine.positive_lp(d,j,s,role='certificate')
                    if fresh is None or fresh['x'][0]<=0:raise RuntimeError('fallback unresolved')
                    lam=float(fresh['x'][0]);q=1/lam;t=-fresh['dual'][-1]
                    if t<=0:raise RuntimeError('fallback witness unavailable')
                    p=ctx.point(-fresh['dual'][:-1]/t,j);v=fresh['x'][1:]/lam
                else:
                    fresh=engine.negative_lp(d,j,s,role='certificate')
                    if fresh is None or fresh['x'][-1]<=0:raise RuntimeError('fallback unresolved')
                    gamma=-float(fresh['dual'][-1])
                    if gamma<=0:raise RuntimeError('fallback gamma unavailable')
                    q=-1/gamma;p=ctx.point(fresh['x'][:-1]/fresh['x'][-1],j)
                    v=fresh['dual'][:-1]/gamma
                ql=max(ql,float(p['qlo'][j]));qs,L,corr=ctx.coefficient(q,v,j,delta)
            if qs<ql or qs-ql>engine.TARGET*(1+abs(qs)):
                raise RuntimeError('coefficient-quality-unresolved')
            qcert=ctx.last_certificate['q_nom']
            if qcert==0:raise RuntimeError('zero nominal coefficient: distance unavailable')
            r['rays'].append(dict(j=j,branch=branch,q_safe=qs,q_lower=ql,q_width=qs-ql,
                                 raw_distance=abs(1/qcert),coefficient_correction=corr,
                                 certificate_lower_before_correction=L,
                                 certificate_status='interval_verified'))
            proofs.append(dict(**ctx.last_certificate,q_lower=ql,j=j,
                               lower_witness=ctx.lower_witness[j].copy()))
        if all(x['branch']=='negative' for x in r['rays']):r['state']='no-improvement'
    except Exception as exc:
        r['state']='timeout' if isinstance(exc,TimeoutError) or 'Time limit' in str(exc) else 'failed'
        r['failure_reason']=str(exc)
    r['wall_total']=time.perf_counter()-start
    if origin_proof is not None:
        r['_proof']=dict(origin_dual=origin_proof,origin_delta_lower=delta,
                        ray_indices=np.array([p['j'] for p in proofs]),
                        cert_q_nom=np.array([p['q_nom'] for p in proofs]),
                        cert_dual=np.array([p['dual'] for p in proofs]),
                        q_safe=np.array([p['q_safe'] for p in proofs]),
                        q_lower=np.array([p['q_lower'] for p in proofs]),
                        lower_before=np.array([p['lower_before'] for p in proofs]),
                        lower_witness=np.array([p['lower_witness'] for p in proofs]))
    r.update(s.times)
    r.update(lp_calls=s.lp_calls,simplex_iterations=s.pivots,warm_calls=s.warm_calls,
             zero_pivot_warm_calls=s.zero_pivot_warm_calls,pool_size_peak=0,pool_bound_improved=0,
             oracle_calls=s.oracle_calls,classification_calls=s.classification_calls,
             origin_calls=s.origin_calls,direct_calls=s.direct_calls,polish_calls=s.polish_calls,
             fallback_calls=s.fallback_calls,model_warnings=s.model_warnings,
             dropped_coefficients=s.dropped_coefficients,
             max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    return r
