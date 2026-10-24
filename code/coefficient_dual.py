"""Fixed common-coefficient dual LP, RHS updates and retained basis.

Adapters expose the equivalent A-denominator primal/dual solution to the
unchanged, certified Q implementation. Timing includes both constructions.
"""
import inspect

import highspy
import numpy as np
from scipy import sparse

import engine
from common_coefficient import run_q

BASE_LP=engine.LP

class CoefficientDualLP(BASE_LP):
    def __init__(self,matrix,rhs,cost,stats,**kwargs):
        caller=inspect.currentframe().f_back
        self.dual_mode=(caller.f_code.co_name=='run_q' and kwargs.get('row_lower') is not None)
        if self.dual_mode:
            d=caller.f_locals['d'];rows=len(d.h)
            transformed=sparse.vstack([
                sparse.hstack([sparse.csr_matrix((-d.a)[:,None]),d.G.T]),
                sparse.csr_matrix(np.r_[-d.a0,-d.h][None,:])],format='csc')
            self.row_indices=np.arange(len(d.a)+1,dtype=np.int32)
            self.row_lower=np.full(len(d.a)+1,-engine.INF)
            super().__init__(transformed,np.asarray(cost),np.r_[1.,np.zeros(rows)],stats,
                             col_lower=np.full(rows+1,-engine.INF),
                             col_upper=np.r_[engine.INF,np.zeros(rows)])
        else:super().__init__(matrix,rhs,cost,stats,**kwargs)

    def solve(self,cost=None,role='oracle',**kwargs):
        if not self.dual_mode:return super().solve(cost,role=role,**kwargs)
        if cost is not None:
            status=self.h.changeRowsBounds(len(self.row_indices),self.row_indices,
                                           self.row_lower,np.asarray(cost))
            if status!=highspy.HighsStatus.kOk:raise RuntimeError('RHS update failed')
        r=super().solve(None,role=role,**kwargs)
        if r is None:return None
        return dict(x=-r['dual'],dual=np.r_[r['x'][1:],-r['x'][0]],value=-r['value'])

def run_qd(d):
    original=engine.LP;engine.LP=CoefficientDualLP
    try:r=run_q(d)
    finally:engine.LP=original
    r['method']='QD'
    return r
