"""Actual batch inference after frozen selection, on the existing ToN dev tail.

Primary unit: current observed record. No retrospective window attribution.
Arrival proxy, batching and queue simulation are explicit; not live export timing.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
from time import perf_counter,process_time
from collections import defaultdict
import json,pickle,hashlib,datetime
import numpy as np,pandas as pd,joblib,psutil
from threadpoolctl import threadpool_limits
from behavioral_data import META_COLUMNS,metadata,event_columns,META_DTYPE
from behavioral_models import scores
from behavioral_observation import BehavioralObservation
from run_behavioral_comparison import R,D,O,inputs,evaluate_holdouts,load_part
from run_network_context_study import slice_data
from run_multienvironment_training import ton_temporal_boundaries
from multienvironment_features import transform

RAW_FIELDS=['PROTOCOL','FLOW_DURATION_MILLISECONDS','IN_PKTS','OUT_PKTS','IN_BYTES','OUT_BYTES',
            'RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_PKTS','SHORTEST_FLOW_PKT','LONGEST_FLOW_PKT','TCP_FLAGS']
def empty():return dict(benign=0,attacks=0,fp=0,tp=0)
def update(c,y,p):
    c['benign']+=int((y==0).sum());c['attacks']+=int((y==1).sum());c['fp']+=int(p[y==0].sum());c['tp']+=int(p[y==1].sum())
def finish(c):return dict(**c,recall=c['tp']/c['attacks'] if c['attacks'] else None,fpr=c['fp']/c['benign'] if c['benign'] else None,
    precision=c['tp']/max(1,c['tp']+c['fp']),f1=2*c['tp']/max(1,2*c['tp']+c['fp']+c['attacks']-c['tp']))

class Histogram:
    def __init__(self):self.edges=np.r_[0.,np.geomspace(.0001,1e9,4096),np.inf];self.counts=np.zeros(len(self.edges),np.int64);self.n=0;self.total=0.;self.maximum=0.
    def add(self,values):
        v=np.asarray(values);assert (v>=0).all();self.counts+=np.bincount(np.searchsorted(self.edges,v,side='left'),minlength=len(self.edges));self.n+=len(v);self.total+=float(v.sum());self.maximum=max(self.maximum,float(v.max(initial=0)))
    def summary(self):
        cumulative=np.cumsum(self.counts)
        return dict(n=self.n,mean_ms=self.total/max(1,self.n),max_ms=self.maximum,
            **{name:float(self.edges[min(len(self.edges)-1,int(np.searchsorted(cumulative,q*self.n)))]) if self.n else None for name,q in [('p50_upper_ms',.5),('p95_upper_ms',.95),('p99_upper_ms',.99)]},
            scope='Log-bin upper-edge quantiles; exact mean/max. Simulated receipt-to-sink time from measured component service, excludes real exporter/transport.')

def main():
    saved=json.loads((O/'selections.json').read_text());selected=[]
    for c in saved:
        if c['seed']!=42:continue
        s=c['selected']['shared_pooled'];selected.append(dict(name=c['method'],method=c['method'],artifact=s['artifact'],threshold=s['thresholds']['shared'],policy='shared_pooled'))
    flow=next(c for c in saved if c['method']=='flow14' and c['seed']==42)['selected']['shared_window']
    selected.append(dict(name='flow14_reference',method='flow14',artifact=flow['artifact'],threshold=flow['thresholds']['shared'],policy='shared_window'))
    models={s['artifact']:joblib.load(R/s['artifact']) for s in selected}
    for s in selected:s['artifact_sha256']=hashlib.sha256((R/s['artifact']).read_bytes()).hexdigest()
    (O/'full_replay_protocol.json').write_text(json.dumps(dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),
        selected=selected,seed=42,batch_max=1000,wait_deadline_ms=1000,flow_window=10000,time_window_ms=60000,
        prediction_unit='Current completed record only',arrival='Ordered completion time proxy, not real exporter or sensor receipt timestamps.',
        timing='Actual observation/native model/in-memory sink costs; counterfactual per-method serial queue and1s/max1000 batching. No measured live exporter/transport.',
        episodes='Evaluator-only source/family attack-record bursts separated by >60s inactivity. Left-censor initial60s. Never reassign a benign or earlier flow as a detection.'),indent=2))
    with (O/'nf_prefix_state.pkl').open('rb') as f:engine=pickle.load(f)
    assert isinstance(engine,BehavioralObservation)
    prefix=json.loads((D/'manifest.json').read_text())['boundaries'][2];start,stop=json.loads((D/'manifest.json').read_text())['boundaries'][-2:]
    with np.load(D/'evaluation.npz') as z:probe_ids=z['row'].copy()
    probe_index={int(row):j for j,row in enumerate(probe_ids)};probe=np.zeros((len(probe_ids),26),np.float32);probe_meta=np.empty(len(probe_ids),META_DTYPE);seen=np.zeros(len(probe_ids),bool)
    totals={s['name']:empty() for s in selected};flow_windows={s['name']:defaultdict(empty) for s in selected};time_windows={s['name']:defaultdict(empty) for s in selected}
    families={s['name']:defaultdict(lambda:dict(n=0,tp=0)) for s in selected};latency={s['name']:Histogram() for s in selected};alert_latency={s['name']:Histogram() for s in selected}
    servers={s['name']:0. for s in selected};episodes=[];active_episode={};buffer=[];first_t=None;frontend=0.;observation_ms=[];batch_compute=[];batch_sizes=[];source_offset=0;observed=0;processed=0
    wall=perf_counter();cpu=process_time()

    def episode_for(source,family,t,flow_start):
        key=(source,family);previous=active_episode.get(key)
        if previous is None or t-episodes[previous]['last']>60000:
            previous=len(episodes);episodes.append(dict(source=int(source),family=str(family),first=int(t),last=int(t),first_flow_start=int(flow_start),
                left_censored=bool(t-first_t<=60000),detections={s['name']:None for s in selected}));active_episode[key]=previous
        episodes[previous]['last']=int(t);return previous

    def flush(release):
        nonlocal buffer
        if not buffer:return
        tick=perf_counter();yy=np.array([b['y'] for b in buffer],np.uint8);ff=np.array([b['family'] for b in buffer]);tt=np.array([b['t'] for b in buffer],np.float64)
        beh=np.array([b['behavior'] for b in buffer],np.float32);raw=np.array([b['raw'] for b in buffer],np.float32)
        prep=perf_counter();physical=transform(raw,RAW_FIELDS,'physical14');physical_ms=(perf_counter()-prep)*1000
        fw=np.array([(b['row']-start)//10000 for b in buffer]);tw=((tt-first_t)//60000).astype(int);ready=max(b['ready'] for b in buffer)
        prediction_cache={}
        for s in selected:
            name=s['name'];method=s['method'];service_start=perf_counter()
            if method=='flow14':x=physical
            elif method=='hybrid38':x=np.column_stack([physical,beh[:,:24]])
            elif method=='behavior26_global_ablation':x=beh
            else:x=np.ascontiguousarray(beh[:,:24])
            cache_key=(s['artifact'],method)
            reused_prediction=cache_key in prediction_cache
            if reused_prediction:score,predict_ms=prediction_cache[cache_key]
            else:
                before=perf_counter();score=scores(models[s['artifact']],x);predict_ms=(perf_counter()-before)*1000;prediction_cache[cache_key]=(score,predict_ms)
            pred=score>s['threshold'];positive_indices=np.flatnonzero(pred) #common in-memory alert sink
            assembly_and_sink_ms=(perf_counter()-service_start)*1000
            # Reused calls still pay their own measured native service in each
            # hypothetical deployment. Physical preprocessing belongs to flow/hybrid.
            service_ms=assembly_and_sink_ms+(predict_ms if reused_prediction else 0.)+(physical_ms if method in ['flow14','hybrid38'] else 0.)
            earliest=tt[-1] if method=='flow14' else ready
            delivered=max(float(release),servers[name],earliest)+service_ms;servers[name]=delivered
            delay=np.maximum(0,delivered-tt);latency[name].add(delay);alert_latency[name].add(delay[pred])
            update(totals[name],yy,pred)
            for w in np.unique(fw):mask=fw==w;update(flow_windows[name][int(w)],yy[mask],pred[mask])
            for w in np.unique(tw):mask=tw==w;update(time_windows[name][int(w)],yy[mask],pred[mask])
            for fam in np.unique(ff[yy==1]):
                mask=(ff==fam)&(yy==1);families[name][str(fam)]['n']+=int(mask.sum());families[name][str(fam)]['tp']+=int(pred[mask].sum())
            for j in positive_indices:
                if yy[j] and episodes[buffer[j]['episode']]['detections'][name] is None:episodes[buffer[j]['episode']]['detections'][name]=delivered
        batch_compute.append((perf_counter()-tick)*1000);batch_sizes.append(len(buffer));buffer=[]

    columns=list(dict.fromkeys(META_COLUMNS+RAW_FIELDS+['FLOW_START_MILLISECONDS','Label','Attack']))
    for frame in pd.read_csv(D.parent/'NF-ToN-IoT-v3_endtime_ordered.csv',usecols=columns,chunksize=25000):
        end=source_offset+len(frame)
        if end<=prefix:source_offset=end;continue
        take=frame.iloc[max(0,prefix-source_offset):min(len(frame),stop-source_offset)]
        base=max(prefix,source_offset);m=metadata(take,base);raw=take[RAW_FIELDS].to_numpy(dtype=np.float32)
        labels=take.Label.to_numpy();assert np.isin(labels,[0,1]).all();fam=take.Attack.to_numpy(dtype=str);starts=take.FLOW_START_MILLISECONDS.to_numpy(dtype=np.int64)
        for i,event in enumerate(zip(*event_columns(m))):
            row=int(m['row'][i]);t=int(m['t'][i]);query=row>=start
            if query and first_t is None:first_t=t;frontend=float(t)
            if query and buffer and t>=buffer[0]['t']+1000:flush(buffer[0]['t']+1000)
            tick=perf_counter();v=engine.observe(*event,query=query)
            if query:feature=(*v,*engine.global_features())
            cost=(perf_counter()-tick)*1000;observed+=1
            if not query:continue
            frontend=max(float(t),frontend)+cost
            if len(observation_ms)<1000000:observation_ms.append(cost)
            if row in probe_index:
                j=probe_index[row];probe[j]=feature;probe_meta[j]=m[i];seen[j]=True
            y=int(labels[i]);ep=episode_for(int(m['source'][i]),fam[i],t,int(starts[i])) if y else None
            buffer.append(dict(row=row,t=t,ready=frontend,raw=raw[i],behavior=feature,y=y,family=fam[i],episode=ep))
            processed+=1
            if len(buffer)>=1000:flush(t)
        source_offset=end
        if processed and processed//1000000!=(processed-len(take))//1000000:print('behavioral full replay',processed,'elapsed',round(perf_counter()-wall,1),flush=True)
        if end>=stop:break
    if buffer:flush(buffer[0]['t']+1000)
    assert processed==stop-start and seen.all()
    np.save(O/'ton_evaluation_behavior.npy',probe);np.save(O/'ton_evaluation_metadata.npy',probe_meta)
    result=[]
    for s in selected:
        name=s['name'];rr=dict(**s,**finish(totals[name]),family={f:dict(**c,recall=c['tp']/c['n']) for f,c in families[name].items()},latency=latency[name].summary(),alert_latency=alert_latency[name].summary())
        for unit,table in [('flow',flow_windows[name]),('time',time_windows[name])]:
            ws=[dict(window=w,**finish(c)) for w,c in sorted(table.items())]
            for key in ['tp','fp','attacks','benign']:assert sum(w[key] for w in ws)==totals[name][key]
            rr[unit+'_windows']=ws;rr['worst_supported_'+unit+'_window_fpr']=max(w['fpr'] for w in ws if w['benign']>=100)
        eligible=[e for e in episodes if not e['left_censored']];detected=[e for e in eligible if e['detections'][name] is not None]
        delays=[e['detections'][name]-e['first'] for e in detected]
        rr['episode_detection']=dict(uncensored_episodes=len(eligible),detected=len(detected),censored_episodes=len(episodes)-len(eligible),
            median_receipt_to_detection_ms=float(np.median(delays)) if delays else None,p95_receipt_to_detection_ms=float(np.quantile(delays,.95)) if delays else None,
            scope='Source/family bursts after60s inactivity, not externally verified incidents. Undetected episodes excluded from delay quantiles and counted separately.')
        result.append(rr)
    resources=dict(wall_s=perf_counter()-wall,cpu_s=process_time()-cpu,observed_records=observed,scored_records=processed,batches=len(batch_sizes),
        batch_sizes=dict(min=min(batch_sizes),max=max(batch_sizes)),actual_combined_batch_p95_ms=float(np.quantile(batch_compute,.95)),
        observation_first_million_p95_ms=float(np.quantile(observation_ms,.95)),observation_first_million_mean_ms=float(np.mean(observation_ms)),
        peak_resident_bytes=psutil.Process().memory_info()._asdict().get('peak_wset'),state=engine.diagnostics(),
        scope='Actual offline desktop run, includes CSV scan. Queue/alert quantiles simulate idealized completion-time arrivals using measured component costs; not live export or network delivery.')
    (O/'full_replay_results.json').write_text(json.dumps(dict(resources=resources,results=result),indent=2))
    #Public report omits raw endpoint IDs, retaining only aggregate episode data.
    (O/'episode_counts.json').write_text(json.dumps(dict(total=len(episodes),left_censored=sum(e['left_censored'] for e in episodes),
        families={str(f):sum(e['family']==f for e in episodes) for f in {e['family'] for e in episodes}}),indent=2))
    print('COMPLETE full behavioral replay',resources,flush=True)
    tune=load_part(D,'tune','ton');_,b=ton_temporal_boundaries();hold={'ton_future':slice_data(tune,slice(b,None)),
        'unsw_future':load_part(__import__('run_network_context_study').U,'development_holdout','unsw'),'ton_stress':load_part(D,'evaluation','ton')}
    evaluate_holdouts(saved,hold,O/'results.json')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
