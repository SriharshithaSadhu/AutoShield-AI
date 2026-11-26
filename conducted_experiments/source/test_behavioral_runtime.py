"""Real-prefix integration parity; no protected flows or synthetic evaluation."""
import json,unittest
import numpy as np,pandas as pd,joblib
from behavioral_runtime import BehavioralRuntime,CompletedFlow,RAW_FIELDS
from behavioral_observation import BehavioralObservation
from behavioral_data import META_COLUMNS,metadata,event_columns
from behavioral_models import scores
from multienvironment_features import transform
from run_behavioral_comparison import R,O

class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame=pd.read_csv(R/'data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv',usecols=list(dict.fromkeys(META_COLUMNS+list(RAW_FIELDS))),nrows=2500)
        cls.m=metadata(frame,0);cls.raw=frame[list(RAW_FIELDS)].to_numpy(dtype=np.float32)
        cls.records=[CompletedFlow(int(e[0]),int(e[1]),int(e[2]),int(e[3]),tuple(cls.raw[i])) for i,e in enumerate(zip(*event_columns(cls.m)))]
        cls.choices=[c for c in json.loads((O/'selections.json').read_text()) if c['seed']==42]

    def test_real_trace_batch_partition_does_not_change_scores(self):
        for c in self.choices:
            s=c['selected']['shared_pooled'];model=joblib.load(R/s['artifact']);method=c['method'];a=BehavioralRuntime(model,method,s['thresholds']['shared'],max_batch=2500);b=BehavioralRuntime(model,method,s['thresholds']['shared'])
            expected=a.score_batch(self.records);parts=[b.score_batch(self.records[i:i+137]) for i in range(0,len(self.records),137)]
            np.testing.assert_allclose(expected['scores'],np.concatenate([p['scores'] for p in parts]),rtol=0,atol=1e-14)
            np.testing.assert_equal(expected['alerts'],np.concatenate([p['alerts'] for p in parts]))
            # Independent reference engine plus the frozen offline transform.
            engine=BehavioralObservation();values=[]
            for r in self.records:
                v=engine.observe(*r.event());values.append((*v,*engine.global_features()))
            beh=np.asarray(values,np.float32)
            physical=transform(self.raw,list(RAW_FIELDS),'physical14')
            x=physical if method=='flow14' else np.column_stack([physical,beh[:,:24]]) if method=='hybrid38' else beh if method=='behavior26_global_ablation' else np.ascontiguousarray(beh[:,:24])
            np.testing.assert_allclose(expected['scores'],scores(model,x),rtol=0,atol=1e-14)

    def test_bad_order_rejected_before_state_mutation(self):
        c=next(c for c in self.choices if c['method']=='hybrid38')['selected']['shared_pooled'];r=BehavioralRuntime(joblib.load(R/c['artifact']),'hybrid38',c['thresholds']['shared'])
        first=self.records[0];bad=CompletedFlow(first.arrival_ms-1,first.source,first.destination,first.destination_port,first.raw)
        with self.assertRaises(ValueError):r.score_batch([first,bad])
        self.assertEqual(r.observation.records,0);self.assertIsNone(r.last_arrival)
        with self.assertRaises(ValueError):r.score_batch(self.records)
        self.assertEqual(r.observation.records,0)

    def test_label_free_record_contract(self):
        self.assertEqual(set(CompletedFlow.__dataclass_fields__),{'arrival_ms','source','destination','destination_port','raw'})

if __name__=='__main__':unittest.main()
