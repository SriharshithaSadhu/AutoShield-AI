import itertools
import unittest
import numpy as np
from audit_context_information_bound import information_bound

class InformationBoundTests(unittest.TestCase):
    def test_matches_exhaustive_vector_decisions(self):
        rng=np.random.default_rng(7)
        for _ in range(30):
            x=rng.integers(0,5,size=18);y=rng.integers(0,2,size=18)
            if not y.sum() or y.sum()==len(y):continue
            cap=.25;budget=int(np.floor(cap*(y==0).sum()));best=0
            for bits in itertools.product([0,1],repeat=5):
                pred=np.asarray(bits)[x]
                if pred[y==0].sum()<=budget:best=max(best,int(pred[y==1].sum()))
            result=information_bound(x[:,None].astype(np.float32),y,cap)
            self.assertEqual(result['label_assisted_optimal_tp'],best)

    def test_nan_values_form_one_vector(self):
        x=np.array([[np.nan],[np.nan],[1],[1]],dtype=np.float32);y=np.array([0,1,1,1])
        result=information_bound(x,y,cap=0.)
        self.assertEqual(result['distinct_vectors'],2)
        self.assertEqual(result['label_assisted_optimal_tp'],2)
        self.assertEqual(result['mixed_vectors'],1)

if __name__=='__main__':unittest.main()
