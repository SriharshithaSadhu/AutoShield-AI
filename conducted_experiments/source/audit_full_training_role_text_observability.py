"""Role-complete raw-text audit of the fixed legitimate ToN training prefix.

Preserves original46 fields PLUS service category AND both service-side hints.
This key is sufficient to reconstruct every frozen contextual representation.
The category-only audit is preserved and is not treated as a complete-role bound.
"""
from collections import defaultdict
from pathlib import Path
from time import perf_counter,process_time
import sqlite3,zlib,json,datetime,shutil,math
import numpy as np,pandas as pd,psutil
from scipy.optimize import milp,Bounds,LinearConstraint
from run_network_context_study import R,D,O
from context_observability_keys import complete_signature
from audit_full_training_text_observability import digest,exact_optimum
B=O/'full_training_role_text_audit'

def main():
    B.mkdir(exist_ok=True);category=json.loads((O/'full_training_text_audit/results.json').read_text());old=category['protocol'];source=R/old['source'];features=old['features'];bound=old['training_prefix_rows']
    script_hashes={name:digest(R/name) for name in ['audit_full_training_role_text_observability.py','context_observability_keys.py','audit_full_training_text_observability.py']}
    p=B/'protocol.json'
    if p.exists():
        protocol=json.loads(p.read_text());assert protocol['code_sha256']==script_hashes
    else:
        protocol=dict(created_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,
            source=old['source'],source_bytes=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,
            training_prefix_rows=bound,exclusive_end_ms=old['exclusive_end_ms'],features=features,
            key='Literal exporter text of46 fields plus category, source-service hint and destination-service hint. Lossless compression after per-chunk aggregation.',
            fpr_cap=.001,chunk_rows=100000,code_sha256=script_hashes,
            correction='Earlier category-only key omitted role side. Earlier outputs remain preserved and are superseded for complete-context interpretation.')
        p.write_text(json.dumps(protocol,indent=2))
    assert source.stat().st_size==protocol['source_bytes'] and source.stat().st_mtime_ns==protocol['source_mtime_ns']
    start_wall=perf_counter();start_cpu=process_time();db=B/'groups.sqlite';conn=sqlite3.connect(db)
    conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA synchronous=NORMAL');conn.execute('PRAGMA cache_size=-65536')
    conn.execute('CREATE TABLE IF NOT EXISTS groups (k BLOB PRIMARY KEY, benign INTEGER NOT NULL, scanning INTEGER NOT NULL) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY CHECK(id=1), source_rows INTEGER, prefix_rows INTEGER, benign INTEGER, scanning INTEGER)')
    conn.execute('INSERT OR IGNORE INTO checkpoint VALUES (1,0,0,0,0)');conn.commit()
    resume,kept,n0,n1=map(int,conn.execute('SELECT source_rows,prefix_rows,benign,scanning FROM checkpoint').fetchone());offset=0
    sql='INSERT INTO groups(k,benign,scanning) VALUES (?,?,?) ON CONFLICT(k) DO UPDATE SET benign=benign+excluded.benign, scanning=scanning+excluded.scanning'
    columns=features+['L4_SRC_PORT','L4_DST_PORT','FLOW_END_MILLISECONDS','Label','Attack']
    if kept<bound:
        for frame in pd.read_csv(source,usecols=columns,dtype=str,keep_default_na=False,chunksize=100000):
            end=offset+len(frame)
            if end<=resume:offset=end;continue
            assert offset>=resume
            t=pd.to_numeric(frame.FLOW_END_MILLISECONDS,errors='raise').to_numpy(dtype=np.int64);d=frame[t<protocol['exclusive_end_ms']];nprefix=len(d)
            label=d.Label.to_numpy();assert np.isin(label,['0','1']).all()
            mask=(label=='0')|((label=='1')&(d.Attack.to_numpy()=='scanning'));d=d.iloc[np.flatnonzero(mask)];y=d.Label.to_numpy()=='1'
            proto=pd.to_numeric(d.PROTOCOL,errors='raise').to_numpy(dtype=float);ports=d[['L4_SRC_PORT','L4_DST_PORT']].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
            signature=complete_signature(proto[:,None],ports,['PROTOCOL']);counts=defaultdict(lambda:[0,0])
            for values,sig,positive in zip(d[features].itertuples(index=False,name=None),signature,y):
                text='|'.join(values);assert text.count('|')==45
                # Exact, injective framing. Compress once per distinct chunk key,
                # rather than paying zlib cost again for every duplicate flow.
                key=text.encode('utf-8')+b'|context='+bytes(sig)
                counts[key][int(positive)]+=1
            if shutil.disk_usage(B).free<1024**3 or db.stat().st_size>2*1024**3:raise RuntimeError('Storage guard; prior checkpoint preserved')
            kept+=nprefix;n0+=int((~y).sum());n1+=int(y.sum());assert kept<=bound
            with conn:
                conn.executemany(sql,[(zlib.compress(k,1),v[0],v[1]) for k,v in counts.items()])
                conn.execute('UPDATE checkpoint SET source_rows=?,prefix_rows=?,benign=?,scanning=? WHERE id=1',(end,kept,n0,n1))
            offset=end
            if offset%2000000==0:print('role-complete raw audit',offset,'source rows;',kept,'training rows',flush=True)
            if kept==bound:break
    assert kept==bound and n0==category['benign'] and n1==category['scanning']
    row=conn.execute('SELECT COUNT(*),SUM(benign),SUM(scanning),SUM(CASE WHEN benign>0 AND scanning>0 THEN 1 ELSE 0 END),SUM(CASE WHEN benign=0 THEN scanning ELSE 0 END) FROM groups').fetchone()
    vectors,b,a,mixed,free=map(int,row);assert b==n0 and a==n1;budget=int(math.floor(.001*n0))
    pairs=conn.execute('SELECT scanning,benign FROM groups WHERE scanning>0 AND benign>0 AND benign<=?',(budget,)).fetchall();av=np.array([z[0] for z in pairs],dtype=np.int64);bv=np.array([z[1] for z in pairs],dtype=np.int64)
    tp=exact_optimum(av,bv,budget,free)
    solver=milp(-av.astype(float),integrality=np.ones(len(av)),bounds=Bounds(0,1),constraints=LinearConstraint(bv[None,:].astype(float),-np.inf,budget),options=dict(time_limit=60,mip_rel_gap=0.))
    independent=dict(success=bool(solver.success),status=int(solver.status),message=solver.message)
    if solver.success:
        chosen=np.rint(solver.x).astype(int);assert int(bv@chosen)<=budget and free+int(av@chosen)==tp
        independent.update(optimal_tp=tp,witness_fp=int(bv@chosen),mip_gap=float(solver.mip_gap))
    assert digest(source,'sha1')==category['source_sha1'];conn.execute('PRAGMA wal_checkpoint(TRUNCATE)');conn.close()
    result=dict(protocol=protocol,source_sha1=category['source_sha1'],publisher_source_sha1_verified=True,
        prefix_rows=kept,benign=n0,scanning=n1,distinct_raw_text_vectors=vectors,mixed_vectors=mixed,free_attack_only_tp=free,
        false_positive_budget=budget,optimal_tp=tp,optimal_scanning_recall=tp/n1,independent_milp=independent,
        resources=dict(wall_s=perf_counter()-start_wall,cpu_s=process_time()-start_cpu,
            peak_resident_bytes=psutil.Process().memory_info()._asdict().get('peak_wset'),database_bytes=db.stat().st_size),
        caveat='Complete frozen contextual input information, no future/history/identity inputs. Optimistic in-sample scanning ceiling; not overall recall infeasibility, annotation-error evidence or a future guarantee.')
    (B/'results.json').write_text(json.dumps(result,indent=2));print('COMPLETE role-complete audit',json.dumps({k:v for k,v in result.items() if k!='protocol'}),flush=True)
if __name__=='__main__':main()
