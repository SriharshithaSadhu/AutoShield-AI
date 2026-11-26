"""Training-only checks: coarse services cannot memorize arbitrary client ports."""
import json
import numpy as np,pandas as pd
from run_network_context_study import D,O,development_inputs
from network_context_features import VARIANTS,SERVICES,transform,service_hints,SWAP_PAIRS

def main():
    train,_,_=development_inputs();x=train['x'];p=train['ports'];f=json.loads((D/'manifest.json').read_text())['features']
    changed=p.copy();protocol=x[:,f.index('PROTOCOL')]
    for side in [0,1]:
        eligible=p[:,side]>=1024
        for proto,mapping in SERVICES.items():eligible&=~((protocol==proto)&np.isin(p[:,side],list(mapping)))
        changed[eligible,side]=60000+side
    port_checks=[]
    for variant in VARIANTS:
        a=transform(x,p,f,variant);b=transform(x,changed,f,variant)
        np.testing.assert_equal(a,b)
        port_checks.append(dict(variant=variant,rows=len(x),unregistered_high_port_replacement_invariant=True))
    hints=service_hints(x,p,f);ids=np.flatnonzero(hints['known']);xx=x[ids].copy();pp=p[ids,::-1]
    for a,b in SWAP_PAIRS:
        ia,ib=f.index(a),f.index(b);xx[:,ia],xx[:,ib]=x[ids,ib],x[ids,ia]
    reverse_checks=[]
    for variant in ['direction15','context29','harmonized_context29']:
        np.testing.assert_equal(transform(x[ids],p[ids],f,variant),transform(xx,pp,f,variant))
        reverse_checks.append(dict(variant=variant,known_role_rows=len(ids),endpoint_direction_reversal_invariant=True))
    registry=pd.read_csv(O/'iana_service_registry_snapshot.csv',dtype=str).fillna('')
    names=[]
    for proto,mapping in SERVICES.items():
        transport='tcp' if proto==6 else 'udp'
        for port,category in mapping.items():
            rr=registry[(registry['Port Number']==str(port))&(registry['Transport Protocol']==transport)]
            assert len(rr), (proto,port)
            names.append(dict(protocol=transport,port=port,category=category,iana_names=rr['Service Name'].tolist()))
    non_tcp=protocol!=6;flags=x[:,f.index('TCP_FLAGS')]
    (O/'identity_invariance_audit.json').write_text(json.dumps(dict(scope=__doc__,
        ports_replaced=int((p!=changed).sum()),port_invariance=port_checks,reverse_invariance=reverse_checks,
        registered_service_entries=names,non_tcp_nonzero_tcp_flags=int((non_tcp&(flags!=0)&np.isfinite(flags)).sum()),
        no_ip_or_absolute_time_input=True,
        caveat='These checks exclude raw client-port identity and direction artifacts. Service distributions can still reveal network environment; endpoint-identity-free is not domain-invariant.'),indent=2))
    print('COMPLETE identity invariance audit',len(x),'rows,',len(ids),'known-role rows')
if __name__=='__main__':main()
