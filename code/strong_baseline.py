"""Exploratory paired check against Newton retaining each paid-for slope basis."""
import argparse
import inspect
import json
from pathlib import Path
import time

import highspy
import numpy as np

import engine
from common_coefficient import run_q
from coefficient_dual import run_qd

ROOT=Path(__file__).resolve().parent
BASE_LP=engine.LP

def engine_frame():
    frame=inspect.currentframe().f_back
    while frame is not None:
        if frame.f_code.co_name=='run_method' and frame.f_globals.get('__name__')=='engine':return frame
        frame=frame.f_back
    return None

class SavedBasisLP(BASE_LP):
    restores=0
    peak=0
    cache_time=0.
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.saved={};self.current_ray=None
    def solve(self,*args,role='oracle',**kwargs):
        frame=engine_frame();j=frame.f_locals.get('j') if frame is not None else None
        start=time.perf_counter()
        if role=='oracle' and j in self.saved and self.current_ray!=j:
            status=self.h.setBasis(self.saved[j])
            if status!=highspy.HighsStatus.kOk:raise RuntimeError('stored basis not accepted')
            self.current_ray=j;SavedBasisLP.restores+=1
        SavedBasisLP.cache_time+=time.perf_counter()-start
        r=super().solve(*args,role=role,**kwargs)
        if role=='classify' and r is not None:
            start=time.perf_counter();basis=self.h.getBasis()
            if not basis.valid:raise RuntimeError('classification basis unavailable')
            self.saved[j]=basis
            SavedBasisLP.peak=max(SavedBasisLP.peak,len(self.saved))
            SavedBasisLP.cache_time+=time.perf_counter()-start
        return r

class NormalizationMixin:
    def solve(self,cost=None,role='oracle',**kwargs):
        caller=engine_frame()
        j=caller.f_locals.get('j') if caller is not None else None
        scale=None
        if role=='oracle' and caller is not None:
            d=caller.f_locals['d'];q=caller.f_locals['q']
            scale=caller.f_locals['distance']
            cost=d.B[j]+q*d.a
        r=super().solve(cost,role=role,**kwargs)
        if r is not None and scale is not None:
            # The frozen engine expects the unscaled oracle's dual. Its
            # subsequent division reconstructs a sign-valid nominal dual;
            # the independent residual certificate handles rounding losses.
            r['dual']=r['dual']*scale;r['value']=r['value']*scale
        return r

class NormalizedLP(NormalizationMixin,BASE_LP):pass
class NormalizedSavedLP(NormalizationMixin,SavedBasisLP):pass

class PrimalOracleMixin:
    def solve(self,cost=None,role='oracle',**kwargs):
        # Y's matrix/bounds are fixed, so objective changes preserve primal
        # basis feasibility. Compare explicit primal reoptimization as well.
        status=self.h.setOptionValue('simplex_strategy',4 if role=='oracle' else 1)
        if status!=highspy.HighsStatus.kOk:raise RuntimeError('simplex strategy unavailable')
        return super().solve(cost,role=role,**kwargs)

class PrimalNormalizedLP(PrimalOracleMixin,NormalizedLP):pass
class PrimalNormalizedSavedLP(PrimalOracleMixin,NormalizedSavedLP):pass
class PrimalCoefficientLP(PrimalOracleMixin,BASE_LP):pass

def run(d,method):
    SavedBasisLP.restores=SavedBasisLP.peak=0;SavedBasisLP.cache_time=0.
    engine.LP={'N_saved_basis':SavedBasisLP,'N_norm':NormalizedLP,
               'N_saved_norm':NormalizedSavedLP,'N_primal_norm':PrimalNormalizedLP,
               'N_primal_saved_norm':PrimalNormalizedSavedLP,'Q_primal':PrimalCoefficientLP}.get(method,BASE_LP)
    try:r=(run_qd(d) if method=='QD' else run_q(d) if method.startswith('Q') else engine.run_method(d,'N'))
    finally:engine.LP=BASE_LP
    r.update(method=method,basis_restore_calls=SavedBasisLP.restores,
             stored_basis_count_peak=SavedBasisLP.peak,basis_cache_time_s=SavedBasisLP.cache_time)
    return r
