"""Exact rational LP relaxations for empirical per-vector observability audits.

Not statistical generalization bounds or a new Neyman-Pearson algorithm.
"""
from fractions import Fraction

def rank_groups(attack,benign):
    pairs=[(int(a),int(b)) for a,b in zip(attack,benign) if int(a)>0 and int(b)>0]
    return sorted(pairs,key=lambda z:Fraction(z[0],z[1]),reverse=True)

def upper_tp(attack,benign,budget,free=0):
    pairs=rank_groups(attack,benign);remaining=Fraction(budget);value=Fraction(free);slope=Fraction(0)
    for a,b in pairs:
        take=min(remaining,Fraction(b));value+=Fraction(a,b)*take;remaining-=take
        if remaining==0:
            slope=Fraction(a,b);break
    # Lagrangian weak duality: a*z <= lambda*b*z + max(0,a-lambda*b).
    dual=Fraction(free)+slope*budget+sum(max(Fraction(0),Fraction(a)-slope*b) for a,b in pairs)
    assert dual==value
    return value,slope,dual

def lower_fp_for_tp(attack,benign,target,free=0):
    remaining=max(Fraction(0),Fraction(target)-free);cost=Fraction(0)
    for a,b in rank_groups(attack,benign):
        take=min(remaining,Fraction(a));cost+=Fraction(b,a)*take;remaining-=take
        if remaining==0:return cost
    if remaining:raise ValueError('Target exceeds total attack count')
    return cost

def rational(x):return dict(numerator=x.numerator,denominator=x.denominator,value=float(x))
