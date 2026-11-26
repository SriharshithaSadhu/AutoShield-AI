import unittest
from audit_behavioral_false_alarm_context import PriorAttacks

class PriorAttackTests(unittest.TestCase):
    def test_same_time_and_future_cannot_explain_false_alarm(self):
        p=PriorAttacks();self.assertEqual(p.before(1,10),-1);p.add(1,10)
        self.assertEqual(p.before(1,10),-1);self.assertEqual(p.before(1,11),10)
        p.add(1,20);p.add(1,20);self.assertEqual(p.before(1,20),10)
        self.assertEqual(p.before(1,21),20)
        with self.assertRaises(ValueError):p.add(1,19)
if __name__=='__main__':unittest.main()
