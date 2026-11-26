"""Reproduce supplementary precision and short-flow conflict diagnostics."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation';O=R/'results/detection_foundation'
def main():
 features=json.loads((D/'manifest.json').read_text())['features']
 with np.load(D/'raw_conflict_training.npz') as z:x=z['x'];y=z['y']
 delta=np.abs(x-x.astype(np.float32).astype(np.float64));changed=delta>0
 precision=dict(rows=len(y),rows_with_any_precision_change=int(changed.any(axis=1).sum()),features_with_precision_change={features[i]:dict(changed_rows=int(changed[:,i].sum()),max_absolute_difference=float(delta[:,i].max())) for i in range(len(features)) if changed[:,i].any()},scope='Raw training conflict rows only; float32 conversion explains none of their conflicts.')
 def col(name):return x[:,features.index(name)]
 patterns=dict(total_rows=len(y),one_packet_unidirectional=int(((col('IN_PKTS')==1)&(col('OUT_PKTS')==0)).sum()),at_most_two_total_packets=int(((col('IN_PKTS')+col('OUT_PKTS'))<=2).sum()),one_ms_duration=int((col('FLOW_DURATION_MILLISECONDS')==1).sum()),zero_duration=int((col('FLOW_DURATION_MILLISECONDS')==0).sum()),syn_only=int((col('TCP_FLAGS')==2).sum()),scope='Training-only patterns; these do not establish annotation errors.')
 for name,value in [('float32_precision_changes',precision),('conflict_flow_patterns',patterns)]: (O/(name+'.json')).write_text(json.dumps(value,indent=2))
 print(precision);print(patterns)
if __name__=='__main__':main()
