import unittest
import numpy as np
from context_observability_keys import complete_signature
from test_network_context_features import flow,FEATURES
from network_context_features import transform

class CompleteKeyTests(unittest.TestCase):
    def test_category_alone_loses_endpoint_role(self):
        x=np.repeat(flow(),2,axis=0);ports=np.array([[50000,443],[443,50000]])
        sig=complete_signature(x,ports,FEATURES)
        self.assertEqual(sig[0,0],sig[1,0])
        self.assertFalse(np.array_equal(sig[0],sig[1]))
        self.assertFalse(np.array_equal(transform(x,ports,FEATURES,'direction15')[0],
                                       transform(x,ports,FEATURES,'direction15')[1]))

    def test_client_port_changes_preserve_complete_signature(self):
        x=np.repeat(flow(),2,axis=0);sig=complete_signature(x,np.array([[50000,443],[60000,443]]),FEATURES)
        np.testing.assert_equal(sig[0],sig[1])

if __name__=='__main__':unittest.main()
