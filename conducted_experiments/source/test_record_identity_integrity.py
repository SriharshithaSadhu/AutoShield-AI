import unittest
from collections import Counter,defaultdict
from audit_record_identity_integrity import inspect
class IdentityTests(unittest.TestCase):
    def test_consistent_and_conflicting_duplicates_counted_separately(self):
        groups={b'a':{(b'0',b'Benign'):[1,2,2,Counter({0:2})]},b'b':{(b'0',b'Benign'):[3,3,1,Counter({0:1})],(b'1',b'xss'):[4,4,1,Counter({1:1})]}}
        c=Counter();e=defaultdict(list);inspect(groups,'complete',c,e)
        self.assertEqual(c['complete_duplicate_rows'],4);self.assertEqual(c['complete_duplicate_groups'],2)
        self.assertEqual(c['complete_binary_conflict_rows'],2);self.assertEqual(c['complete_binary_conflict_partition_1'],1)
if __name__=='__main__':unittest.main()
