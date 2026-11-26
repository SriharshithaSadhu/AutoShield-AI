import unittest
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from observability_relaxation import upper_tp,lower_fp_for_tp

class RelaxationTests(unittest.TestCase):
    def test_exact_primal_dual_and_independent_linear_programs(self):
        rng=np.random.default_rng(91)
        for _ in range(40):
            a=rng.integers(1,20,8);b=rng.integers(1,20,8);budget=int(rng.integers(0,int(b.sum())+1));free=7
            upper,slope,dual=upper_tp(a,b,budget,free)
            ref=linprog(-a,A_ub=b[None,:],b_ub=[budget],bounds=(0,1),method='highs')
            self.assertTrue(ref.success);self.assertAlmostEqual(float(upper),free-ref.fun,places=8);self.assertEqual(upper,dual)
            target=int(rng.integers(free,free+int(a.sum())+1));cost=lower_fp_for_tp(a,b,target,free)
            ref=linprog(b,A_ub=-a[None,:],b_ub=[free-target],bounds=(0,1),method='highs')
            self.assertTrue(ref.success);self.assertAlmostEqual(float(cost),ref.fun,places=8)

    def test_free_attack_groups_need_no_false_positives(self):
        self.assertEqual(lower_fp_for_tp([3],[2],4,free=5),Fraction(0))
        self.assertEqual(upper_tp([3],[2],0,free=5)[0],Fraction(5))

if __name__=='__main__':unittest.main()
