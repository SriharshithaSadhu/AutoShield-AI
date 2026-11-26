import unittest
import numpy as np
from network_context_features import SERVICES,VARIANTS,SWAP_PAIRS,service_hints,transform
from multienvironment_features import transform as original

FEATURES=['PROTOCOL','IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS','TCP_FLAGS',
          'FLOW_DURATION_MILLISECONDS','RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_PKTS',
          'SHORTEST_FLOW_PKT','LONGEST_FLOW_PKT']

def flow(proto=6):
    return np.array([[proto,1000,100,10,2,18,1,1,0,40,500]],dtype=np.float32)

class ContextTests(unittest.TestCase):
    def test_endpoint_reversal_canonicalization(self):
        x=flow();rev=x.copy()
        for a,b in SWAP_PAIRS:
            i,j=FEATURES.index(a),FEATURES.index(b);rev[:,i],rev[:,j]=x[:,j],x[:,i]
        for mode in ['direction15','context29','harmonized_context29']:
            np.testing.assert_allclose(transform(x,np.array([[50000,443]]),FEATURES,mode),
                                       transform(rev,np.array([[443,50000]]),FEATURES,mode))

    def test_ambiguous_service_does_not_force_direction(self):
        x=np.repeat(flow(),3,axis=0);p=np.array([[53,443],[100,443],[50000,50001]])
        h=service_hints(x,p,FEATURES)
        self.assertFalse(h['known'].any())
        np.testing.assert_equal(transform(x,p,FEATURES,'direction15')[:,:14],original(x,FEATURES,'physical14'))

    def test_transport_specific_mapping(self):
        h=service_hints(flow(17),np.array([[50000,22]]),FEATURES)
        self.assertFalse(h['known'][0])
        h=service_hints(flow(17),np.array([[50000,53]]),FEATURES)
        self.assertTrue(h['destination_service_hint'][0]);self.assertEqual(h['category'][0],3)

    def test_shapes_and_identity_free_input(self):
        expected=[14,15,26,19,29,14,29]
        for mode,n in zip(VARIANTS,expected):
            self.assertEqual(transform(flow(),np.array([[50000,443]]),FEATURES,mode).shape,(1,n))
        np.testing.assert_equal(transform(flow(),np.array([[50000,443]]),FEATURES,'context29'),
                                transform(flow(),np.array([[60000,443]]),FEATURES,'context29'))

    def test_missing_and_invalid_inputs(self):
        x=flow();x[0,FEATURES.index('IN_BYTES')]=np.nan
        self.assertTrue(np.isnan(transform(x,np.array([[50000,443]]),FEATURES,'context29')[0,3]))
        for ports in [[[50000,-1]],[[50000,70000]],[[50000,2.5]]]:
            with self.assertRaises(ValueError):transform(flow(),np.array(ports),FEATURES,'context29')

if __name__=='__main__':unittest.main()
