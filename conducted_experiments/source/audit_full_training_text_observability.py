"""Exact raw-exporter-text scanning audit on the entire permitted training prefix.

Losslessly compressed SQLite keys avoid retaining the raw corpus in RAM. Model
inputs are the46 permitted aggregate fields plus the already-frozen coarse
service hint. Labels and timestamps select/count audit rows; they are not keys.
No classifier is fitted and no protected UNSW input is opened.
"""
from pathlib import Path
from collections import defaultdict
from time import perf_counter,process_time
import sqlite3,zlib,json,hashlib,datetime,shutil,math
import numpy as np,pandas as pd,psutil
from scipy.optimize import milp,Bounds,LinearConstraint
from run_network_context_study import R,D,O
from network_context_features import service_hints
B=O/'full_training_text_audit'

def digest(path,kind='sha256'):
    h=hashlib.new(kind)
    with path.open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()

def cutoff(bound):
    offset=0
    source=D.parent/'NF-ToN-IoT-v3_endtime_ordered.csv'
    for frame in pd.read_csv(source,usecols=['FLOW_END_MILLISECONDS'],chunksize=500000):
        if offset<=bound<offset+len(frame):
            j=bound-offset;t=frame.FLOW_END_MILLISECONDS.to_numpy(dtype=np.int64)
            assert j>0 and t[j-1]<t[j]
            return int(t[j])
        offset+=len(frame)
    raise ValueError('Training boundary outside ordered source')

def exact_optimum(av,bv,budget,free):
    dp=np.zeros(budget+1,dtype=np.int64)
    for cost in np.unique(bv):
        values=np.sort(av[bv==cost])[::-1][:budget//int(cost)]
        for value in values:dp[cost:]=np.maximum(dp[cost:],dp[:-cost]+value)
    return free+int(dp.max())

def main():
    B.mkdir(exist_ok=True);features=json.loads((D/'manifest.json').read_text())['features'];bound=json.loads((D/'manifest.json').read_text())['boundaries'][1]
    source=D.parent/'NF-ToN-IoT-v3.csv';p=B/'protocol.json';start_wall=perf_counter();start_cpu=process_time()
    if p.exists():
        protocol=json.loads(p.read_text());assert protocol['source_bytes']==source.stat().st_size
        assert protocol['source_mtime_ns']==source.stat().st_mtime_ns
        assert protocol['script_sha256']==digest(Path(__file__)),'Resume with the frozen auditor version.'
    else:
        protocol=dict(created_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,
            source=str(source.relative_to(R)),source_bytes=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,
            training_prefix_rows=bound,exclusive_end_ms=cutoff(bound),features=features,
            key='Original literal exporter text for46 fields, plus fixed coarse category. Pipe separators with verified field count; lossless zlib compression, not a hash.',
            fpr_cap=.001,chunk_rows=100000,script_sha256=digest(Path(__file__)),
            interpretation='Optimistic empirical lookup bound, not a model or future-risk guarantee. Raw text may distinguish numerically equal spellings, making this a conservative upper bound for numeric classifiers.')
        p.write_text(json.dumps(protocol,indent=2))
    db=B/'groups.sqlite';conn=sqlite3.connect(db)
    conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA synchronous=NORMAL');conn.execute('PRAGMA cache_size=-65536')
    conn.execute('CREATE TABLE IF NOT EXISTS groups (k BLOB PRIMARY KEY, benign INTEGER NOT NULL, scanning INTEGER NOT NULL) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY CHECK(id=1), source_rows INTEGER, prefix_rows INTEGER, benign INTEGER, scanning INTEGER)')
    conn.execute('INSERT OR IGNORE INTO checkpoint VALUES (1,0,0,0,0)');conn.commit()
    resumed=conn.execute('SELECT source_rows,prefix_rows,benign,scanning FROM checkpoint WHERE id=1').fetchone()
    resume_offset,kept,benign_total,scanning_total=map(int,resumed);offset=0
    columns=features+['L4_SRC_PORT','L4_DST_PORT','FLOW_END_MILLISECONDS','Label','Attack']
    sql='INSERT INTO groups(k,benign,scanning) VALUES (?,?,?) ON CONFLICT(k) DO UPDATE SET benign=benign+excluded.benign, scanning=scanning+excluded.scanning'
    if kept<bound:
        for frame in pd.read_csv(source,usecols=columns,dtype=str,keep_default_na=False,chunksize=100000):
            end=offset+len(frame)
            if end<=resume_offset:offset=end;continue
            assert offset==resume_offset or offset>resume_offset,'Checkpoint must align with whole source chunks.'
            completion=pd.to_numeric(frame.FLOW_END_MILLISECONDS,errors='raise').to_numpy(dtype=np.int64)
            prefix=frame[completion<protocol['exclusive_end_ms']];n_prefix=len(prefix)
            labels=prefix.Label.to_numpy();assert np.isin(labels,['0','1']).all()
            good=(labels=='0')|((labels=='1')&(prefix.Attack.to_numpy()=='scanning'))
            selected=prefix.iloc[np.flatnonzero(good)];y=(selected.Label.to_numpy()=='1')
            proto=pd.to_numeric(selected.PROTOCOL,errors='raise').to_numpy(dtype=np.float64)
            ports=selected[['L4_SRC_PORT','L4_DST_PORT']].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
            category=service_hints(proto[:,None],ports,['PROTOCOL'])['category']
            counts=defaultdict(lambda:[0,0])
            for values,cat,is_scan in zip(selected[features].itertuples(index=False,name=None),category,y):
                text='|'.join(values);assert text.count('|')==len(features)-1
                key=zlib.compress(text.encode('utf-8')+b'|service='+str(int(cat)).encode('ascii'),1)
                counts[key][int(is_scan)]+=1
            kept+=n_prefix;benign_total+=int((~y).sum());scanning_total+=int(y.sum());assert kept<=bound
            if shutil.disk_usage(B).free<1024**3 or db.stat().st_size>2*1024**3:
                raise RuntimeError('Audit storage guard reached; prior committed checkpoint is preserved.')
            with conn:
                conn.executemany(sql,[(k,v[0],v[1]) for k,v in counts.items()])
                conn.execute('UPDATE checkpoint SET source_rows=?,prefix_rows=?,benign=?,scanning=? WHERE id=1',(end,kept,benign_total,scanning_total))
            offset=end
            if offset%1000000==0:print('raw-text audit source',offset,'training',kept,'eligible',benign_total+scanning_total,'DB MiB',round(db.stat().st_size/1024**2,1),flush=True)
            if kept==bound:break
    assert kept==bound,(kept,bound)
    summary=conn.execute('SELECT COUNT(*),SUM(benign),SUM(scanning),SUM(CASE WHEN benign>0 AND scanning>0 THEN 1 ELSE 0 END),SUM(CASE WHEN benign>0 AND scanning>0 THEN benign+scanning ELSE 0 END),SUM(CASE WHEN benign=0 THEN scanning ELSE 0 END) FROM groups').fetchone()
    vectors,b,a,mixed,mixed_rows,free=map(int,summary);assert b==benign_total and a==scanning_total
    budget=int(math.floor(.001*b));pairs=conn.execute('SELECT scanning,benign FROM groups WHERE scanning>0 AND benign>0 AND benign<=?',(budget,)).fetchall()
    av=np.array([v[0] for v in pairs],dtype=np.int64);bv=np.array([v[1] for v in pairs],dtype=np.int64)
    optimum=exact_optimum(av,bv,budget,free)
    solver=milp(c=-av.astype(float),integrality=np.ones(len(av)),bounds=Bounds(0,1),
        constraints=LinearConstraint(bv[None,:].astype(float),-np.inf,budget),options=dict(time_limit=60,mip_rel_gap=0.))
    independent=dict(success=bool(solver.success),status=int(solver.status),message=solver.message)
    if solver.success:
        chosen=np.rint(solver.x).astype(int);assert int(bv@chosen)<=budget
        assert free+int(av@chosen)==optimum;independent.update(optimal_tp=optimum,witness_fp=int(bv@chosen),mip_gap=float(solver.mip_gap))
    extraction=json.loads((D.parent/'extraction_manifest.json').read_text());expected=next(v['sha1_bagit'] for v in extraction['extracted_and_verified'] if v['path']==source.name)
    actual=digest(source,'sha1');assert actual==expected,'Publisher raw-source hash changed'
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)');conn.close()
    result=dict(protocol=protocol,source_sha1=actual,publisher_source_sha1_verified=True,
        prefix_rows=kept,benign=b,scanning=a,distinct_raw_text_vectors=vectors,mixed_vectors=mixed,rows_in_mixed_vectors=mixed_rows,
        false_positive_budget=budget,free_attack_only_tp=free,optimal_tp=optimum,optimal_scanning_recall=optimum/a,
        independent_milp=independent,raw_source_rows_read=offset,
        resources=dict(wall_s=perf_counter()-start_wall,cpu_s=process_time()-start_cpu,
            peak_resident_bytes=psutil.Process().memory_info()._asdict().get('peak_wset'),database_bytes=db.stat().st_size),
        caveat='Entire legitimate training prefix only, original literal raw-text keys. More permissive than numeric classifiers. Not overall attack recall, annotation-error evidence, independent validation or a future guarantee.')
    (B/'results.json').write_text(json.dumps(result,indent=2));print('COMPLETE full-prefix raw-text audit',json.dumps(result),flush=True)
if __name__=='__main__':main()
