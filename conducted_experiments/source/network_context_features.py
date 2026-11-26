"""Fixed, identity-free service-side hints for completed NetFlow records.

Port registration is a hint, not proof of the application or client/server role.
Definitions are chosen before this study's results; no learned port lists or IPs.
All inputs are available when the exporter completes the flow.
"""
import numpy as np
from multienvironment_features import transform as physical_transform

SERVICE_NAMES = ['web', 'secure_web', 'dns', 'remote', 'mail', 'file',
                 'infrastructure', 'iot', 'directory']
# Protocol numbers: TCP6 / UDP17. Public standard service registrations only.
SERVICES = {
    6: {80:1, 8080:1, 443:2, 53:3, 22:4, 23:4, 3389:4,
        25:5, 110:5, 143:5, 465:5, 587:5, 993:5, 995:5,
        20:6, 21:6, 139:6, 445:6, 1883:8, 8883:8, 389:9, 636:9},
    17: {443:2, 53:3, 3389:4, 69:6, 67:7, 68:7, 123:7,
         161:7, 162:7, 514:7, 389:9}
}
VARIANTS = ['physical14', 'direction15', 'service26', 'protocol19', 'context29',
            'harmonized14', 'harmonized_context29']
SWAP_PAIRS = [('IN_BYTES','OUT_BYTES'), ('IN_PKTS','OUT_PKTS'),
              ('RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_PKTS')]
PROTOCOL_NAMES = ['rst_present','fin_present','psh_present',
                  'tcp_syn_without_ack','udp_without_reverse_packets']


def service_hints(x, ports, features):
    x=np.asarray(x);ports=np.asarray(ports)
    if ports.shape != (len(x),2):raise ValueError('ports must have shape (rows,2)')
    if not np.isfinite(ports).all() or ((ports<0)|(ports>65535)|(ports!=np.floor(ports))).any():
        raise ValueError('invalid transport port')
    protocol=x[:,features.index('PROTOCOL')]
    source=np.zeros(len(x),dtype=np.uint8);dest=np.zeros_like(source)
    for proto,mapping in SERVICES.items():
        mask=protocol==proto
        for port,category in mapping.items():
            source[mask & (ports[:,0]==port)]=category
            dest[mask & (ports[:,1]==port)]=category
    # A low unrecognized port at the other side remains ambiguous.
    src_hint=(source>0)&(dest==0)&(ports[:,1]>=1024)
    dst_hint=(dest>0)&(source==0)&(ports[:,0]>=1024)
    known=src_hint|dst_hint
    category=np.where(src_hint,source,np.where(dst_hint,dest,0))
    return dict(source=source,dest=dest,source_service_hint=src_hint,
                destination_service_hint=dst_hint,known=known,category=category)


def canonicalize(x, features, hints):
    out=np.asarray(x,dtype=np.float32).copy();swap=hints['source_service_hint']
    for a,b in SWAP_PAIRS:
        ia,ib=features.index(a),features.index(b)
        out[swap,ia]=x[swap,ib];out[swap,ib]=x[swap,ia]
    return out


def protocol_context(x, features):
    flags=x[:,features.index('TCP_FLAGS')];proto=x[:,features.index('PROTOCOL')]
    valid=np.isfinite(flags);f=np.where(valid,flags,0).astype(np.int64)
    bits=[((f & k)>0).astype(np.float32) for k in [4,1,8]]
    for b in bits:b[~valid]=np.nan
    halfopen=np.where(valid,((proto==6)&((f&2)>0)&((f&16)==0)).astype(float),np.nan)
    packets=x[:,features.index('OUT_PKTS')]
    udp=np.where(np.isfinite(packets),((proto==17)&(packets==0)).astype(float),np.nan)
    return np.column_stack(bits+[halfopen,udp]).astype(np.float32)


def transform(x, ports, features, variant):
    if variant not in VARIANTS:raise ValueError(variant)
    mode='harmonized14' if variant.startswith('harmonized') else 'physical14'
    if variant in ('physical14','harmonized14'):return physical_transform(x,features,mode)
    hints=service_hints(x,ports,features)
    canonical=variant in ('direction15','context29','harmonized_context29')
    xx=canonicalize(x,features,hints) if canonical else np.asarray(x)
    base=physical_transform(xx,features,mode)
    if variant=='direction15':extra=hints['known'][:,None]
    elif variant=='protocol19':extra=protocol_context(xx,features)
    else:
        onehot=np.column_stack([hints['category']==i for i in range(1,10)])
        if variant=='service26':
            extra=np.column_stack([onehot,hints['source_service_hint'],hints['destination_service_hint'],hints['known']])
        else:extra=np.column_stack([onehot,hints['known'],protocol_context(xx,features)])
    out=np.column_stack([base,extra]).astype(np.float32)
    return np.ascontiguousarray(out)
