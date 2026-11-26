"""Independent raw transport-port and label range audit before accepting replay results."""
import json
import numpy as np,pandas as pd
from run_network_context_study import D,O

def main():
    start,stop=json.loads((D/'manifest.json').read_text())['boundaries'][-2:]
    offset=0;total=0;attacks=0;benign=0;minimum=65535.;maximum=0.
    for frame in pd.read_csv(D.parent/'NF-ToN-IoT-v3_endtime_ordered.csv',usecols=['L4_SRC_PORT','L4_DST_PORT','Label'],chunksize=500000):
        end=offset+len(frame)
        if end<=start:offset=end;continue
        d=frame.iloc[max(0,start-offset):min(len(frame),stop-offset)]
        ports=d[['L4_SRC_PORT','L4_DST_PORT']].to_numpy();y=d.Label.to_numpy()
        assert np.isfinite(ports).all() and ((ports>=0)&(ports<=65535)&(ports==np.floor(ports))).all()
        assert np.isin(y,[0,1]).all()
        total+=len(d);attacks+=int(y.sum());benign+=int((y==0).sum())
        minimum=min(minimum,float(ports.min()));maximum=max(maximum,float(ports.max()));offset=end
        if end>=stop:break
    assert total==stop-start
    (O/'full_input_range_audit.json').write_text(json.dumps(dict(rows=total,attacks=attacks,benign=benign,port_min=minimum,port_max=maximum,
        integer_ports_in_uint16_range=True,binary_labels_before_cast=True,protected_inputs_read=False),indent=2))
    print('COMPLETE independent full-development input range audit',total,'rows')
if __name__=='__main__':main()
