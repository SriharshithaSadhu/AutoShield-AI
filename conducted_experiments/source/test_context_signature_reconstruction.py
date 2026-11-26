import unittest
import numpy as np
from test_network_context_features import flow,FEATURES
from context_observability_keys import complete_signature
from context_from_audit_signature import reconstruct
from network_context_features import transform

class ReconstructionTests(unittest.TestCase):
    def test_matches_frozen_preprocessor_for_roles_protocols_and_missing(self):
        raw=np.concatenate([flow(6),flow(6),flow(17),flow(17),flow(6)])
        raw[-1,FEATURES.index('IN_BYTES')]=np.nan
        ports=np.array([[60000,443],[443,60000],[53,60000],[60000,50000],[53,443]])
        sig=complete_signature(raw,ports,FEATURES)
        for variant in ['physical14','context29']:
            np.testing.assert_equal(reconstruct(raw,sig,FEATURES,variant),transform(raw,ports,FEATURES,variant))
if __name__=='__main__':unittest.main()
