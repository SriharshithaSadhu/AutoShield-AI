"""Bounded exact-text duplicate/label-consistency audit of ordered ToN only.

No training, scoring or protected network access. End-time grouping makes
exact identity checks bounded by one timestamp group rather than corpus size.
"""
from pathlib import Path
from collections import Counter,defaultdict
from time import perf_counter
import hashlib,json,datetime
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3';O=R/'results/integrity_observability_2026-11-26'
BOUNDS=[0,13760005,16060007,18360007,27520260]

def partition(row):return next(i for i in range(4) if BOUNDS[i]<=row<BOUNDS[i+1])

def inspect(groups,kind,counts,examples):
    for key,records in groups.items():
        n=sum(x[2] for x in records.values())
        # A key can repeat with identical labels: records has one item but n>1.
        if n<2:continue
        counts[kind+'_duplicate_groups']+=1;counts[kind+'_duplicate_rows']+=n
        binary={x[0] for x in records};families={x[1] for x in records}
        if len(binary)>1:
            counts[kind+'_binary_conflict_groups']+=1;counts[kind+'_binary_conflict_rows']+=n
            for (label,family),(first,last,number,part_counts) in records.items():
                counts[kind+'_binary_conflict_label_'+label.decode()]+=number
                counts[kind+'_binary_conflict_family_'+family.decode()]+=number
                for part,count in part_counts.items():counts[kind+'_binary_conflict_partition_'+str(part)]+=count
            if len(examples[kind])<20:examples[kind].append(dict(key_sha256=hashlib.sha256(key).hexdigest(),rows=n,labels=[(a.decode(),b.decode(),v[0],v[2]) for (a,b),v in records.items()]))
        if len(families)>1:counts[kind+'_family_conflict_groups']+=1;counts[kind+'_family_conflict_rows']+=n

def main():
    O.mkdir(exist_ok=True);start=perf_counter();counts=Counter();examples=defaultdict(list);exact={};event={};clock=None;previous=None;rows=0;digest=hashlib.sha256();max_group=0;group_size=0
    # Every source field is numeric/IPv4/family text with no comma-bearing values.
    # Reject any unexpected quoting/field count instead of silently parsing it.
    with (D/'NF-ToN-IoT-v3_endtime_ordered.csv').open('rb') as f:
        header=f.readline();digest.update(header);names=header.rstrip(b'\r\n').split(b',');assert len(names)==55
        assert names[:7]==[b'FLOW_START_MILLISECONDS',b'FLOW_END_MILLISECONDS',b'IPV4_SRC_ADDR',b'L4_SRC_PORT',b'IPV4_DST_ADDR',b'L4_DST_PORT',b'PROTOCOL']
        assert names[-2:]==[b'Label',b'Attack']
        for line in f:
            digest.update(line);record=line.rstrip(b'\r\n');fields=record.split(b',');assert len(fields)==55 and b'"' not in record
            t=int(fields[1]);assert previous is None or previous<=t;previous=t
            if clock is not None and t!=clock:
                inspect(exact,'complete_observation',counts,examples);inspect(event,'time_tuple',counts,examples);max_group=max(max_group,group_size);exact={};event={};group_size=0
            clock=t;group_size+=1;label,family=fields[-2:];assert label in (b'0',b'1');part=partition(rows)
            counts['partition_'+str(part)+'_rows']+=1;counts['family_'+family.decode()]+=1
            if (label==b'0')!=(family==b'Benign'):counts['binary_family_disagreement']+=1
            for table,key in [(exact,b','.join(fields[:-2])),(event,b','.join(fields[:7]))]:
                labels=table.setdefault(key,{})
                v=labels.get((label,family))
                if v is None:labels[(label,family)]=[rows,rows,1,Counter({part:1})]
                else:v[1]=rows;v[2]+=1;v[3][part]+=1
            rows+=1
            if rows%5000000==0:print('identity audit rows',rows,'wall',round(perf_counter()-start,1),flush=True)
    inspect(exact,'complete_observation',counts,examples);inspect(event,'time_tuple',counts,examples);max_group=max(max_group,group_size)
    assert rows==27520260;expected=json.loads((R/'results/nf_ton_iot_v3_endtime_ordered_manifest.json').read_text())['output_sha256'];assert digest.hexdigest()==expected
    result=dict(created_at=datetime.datetime.now().astimezone().isoformat(),rows=rows,source_sha256=digest.hexdigest(),wall_s=perf_counter()-start,
        maximum_end_group=max_group,counts=dict(counts),examples=dict(examples),
        complete_observation_key='All53 publisher fields, including tuple and timestamps, excluding onlyLabel/Attack; literal raw CSV field text.',
        time_tuple_key='Start/end plus ordered5tuple; duplicate keys with different remaining fields are not proven duplicate packets.',
        caveat='Consistency audit of exported table only, not validation of packet-level ground truth. No contradictory label is relabeled or removed.')
    (O/'record_identity_integrity.json').write_text(json.dumps(result,indent=2));print('COMPLETE identity integrity',rows,'wall',round(result['wall_s'],1),'counts',dict(counts),flush=True)
if __name__=='__main__':main()
