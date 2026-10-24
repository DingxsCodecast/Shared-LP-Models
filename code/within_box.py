"""Predeclared additional workload with positive endpoints inside X=[0,1]^k.

Engine and solver settings are unchanged. Generation witnesses are saved
separately and are never supplied to any timed method.
"""
import json
from pathlib import Path
import time

import numpy as np

import engine

ROOT=Path(__file__).resolve().parent

def generate(family,rho,seed):
    start=time.perf_counter()
    d=engine.generate(family,128,16,rho,seed)
    stats=engine.Stats();ctx=engine.Context(d,stats)
    lp=engine.LP(d.G,d.h,d.a,stats)
    witnesses=[];indices=[];uppers=[]
    for j,branch in enumerate(d.meta['expected_branches']):
        if branch!='positive':continue
        p=d.a+.5*d.B[j]
        sol=lp.solve(p);y=ctx.feasible(sol['x'])
        _,obj_hi=engine.dot_iv(p,y)
        upper=float(engine.up(d.a0+obj_hi))
        d.b0[j]=float(engine.down(-2*upper-1.))
        # Exact .5 is binary representable. The outward objective bound makes
        # A(y)+.5*B_j(y) <= -.5, proving lambda_plus < .5.
        witnesses.append(y);indices.append(j);uppers.append(-.5)
    d.meta.update(workload='within_box',positive_distance_upper=.5,
                  additional_design='strict improving witness at u=.5 e_j',
                  generation_total_s=time.perf_counter()-start)
    return d,dict(ray_indices=np.array(indices),feasible_y=np.array(witnesses),
                  generation_upper=np.array(uppers))
