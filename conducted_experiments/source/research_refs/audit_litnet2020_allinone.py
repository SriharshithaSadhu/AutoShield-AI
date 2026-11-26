from pathlib import Path
from collections import Counter, defaultdict
import json, time, zipfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'data'/'litnet2020'/'ALLinONE.zip'
OUT=ROOT/'data'/'litnet2020'/'allinone_temporal_audit.json'
FIELDS=['ID','ts_year','ts_month','ts_day','ts_hour','ts_min','ts_second','te_year','te_month','te_day','te_hour','te_min','te_second','td','tr','attack_t','attack_a']

def times(df,prefix):
    return pd.to_datetime({
        'year':pd.to_numeric(df[f'{prefix}_year'],errors='coerce'),
        'month':pd.to_numeric(df[f'{prefix}_month'],errors='coerce'),
        'day':pd.to_numeric(df[f'{prefix}_day'],errors='coerce'),
        'hour':pd.to_numeric(df[f'{prefix}_hour'],errors='coerce'),
        'minute':pd.to_numeric(df[f'{prefix}_min'],errors='coerce'),
        'second':pd.to_numeric(df[f'{prefix}_second'],errors='coerce')
    },errors='coerce',utc=True)

start=time.perf_counter(); rows=0; chunks=0; prev_s=prev_e=prev_id=None; s_desc=e_desc=id_desc=bad_s=bad_e=bad_y=0
smin=smax=emin=emax=None; labels=Counter(); attack_types=Counter(); rec_by_date=defaultdict(Counter); receiver=Counter(); dur_counts=Counter()
with zipfile.ZipFile(ARCHIVE) as z:
    info=z.getinfo('allFlows.csv')
    types={name:pa.string() for name in FIELDS}
    reader=pacsv.open_csv(z.open(info),read_options=pacsv.ReadOptions(block_size=64*1024*1024),convert_options=pacsv.ConvertOptions(include_columns=FIELDS,column_types=types))
    for batch in reader:
        chunks+=1
        df=batch.to_pandas(); n=len(df); rows+=n
        st=times(df,'ts'); et=times(df,'te')
        bad_s+=int(st.isna().sum()); bad_e+=int(et.isna().sum())
        if st.notna().any():
            lo,hi=st.min(),st.max(); smin=lo if smin is None or lo<smin else smin; smax=hi if smax is None or hi>smax else smax
        if et.notna().any():
            lo,hi=et.min(),et.max(); emin=lo if emin is None or lo<emin else emin; emax=hi if emax is None or hi>emax else emax
        sa=st.astype('int64').to_numpy(); ea=et.astype('int64').to_numpy(); sm=st.notna().to_numpy(); em=et.notna().to_numpy()
        if prev_s is not None and sm[0] and sa[0]<prev_s:s_desc+=1
        if prev_e is not None and em[0] and ea[0]<prev_e:e_desc+=1
        s_desc+=int(((sa[1:]<sa[:-1])&sm[1:]&sm[:-1]).sum())
        e_desc+=int(((ea[1:]<ea[:-1])&em[1:]&em[:-1]).sum())
        if sm.any():prev_s=int(sa[np.flatnonzero(sm)[-1]])
        if em.any():prev_e=int(ea[np.flatnonzero(em)[-1]])
        ids=pd.to_numeric(df['ID'],errors='coerce'); bad_y+=int(ids.isna().sum())
        if ids.notna().any():
            if prev_id is not None and int(ids.iloc[0])<prev_id:id_desc+=1
            id_desc+=int((ids.diff()<0).sum())
            prev_id=int(ids.iloc[-1])
        lab=df.attack_a.str.strip(); ty=df.attack_t.str.strip()
        bad_y+=int((~lab.isin(['0','1'])).sum())
        labels.update(lab.value_counts().to_dict()); attack_types.update(ty.value_counts().to_dict())
        receiver.update(df.tr.str.strip().value_counts().to_dict())
        date=st.dt.strftime('%Y-%m-%d').fillna('INVALID')
        for (d,l),cnt in pd.DataFrame({'date':date,'label':lab}).value_counts().items():rec_by_date[str(d)][str(l)]+=int(cnt)
        td=pd.to_numeric(df.td,errors='coerce')
        dur_counts.update({'negative':int((td<0).sum()),'zero':int((td==0).sum()),'missing':int(td.isna().sum())})
        if chunks%10==0: print(f'chunk={chunks} rows={rows:,} elapsed={time.perf_counter()-start:.1f}s',flush=True)
result={'archive_bytes':ARCHIVE.stat().st_size,'lfs_sha256':'443ddfa4e1ce15eeb636ca6847430fd5a6a6d643118d38bfc64eefb56fc9e56b','member':info.filename,'member_uncompressed_bytes':info.file_size,'rows':rows,'chunks':chunks,'start_min_utc':str(smin),'start_max_utc':str(smax),'end_min_utc':str(emin),'end_max_utc':str(emax),'start_adjacent_descents':s_desc,'end_adjacent_descents':e_desc,'invalid_timestamps':{'start':bad_s,'end':bad_e},'invalid_id_or_label':bad_y,'binary_label_counts':dict(labels),'attack_type_counts':dict(attack_types),'date_label_counts':{d:dict(c) for d,c in sorted(rec_by_date.items())},'receiver_time_distinct':len(receiver),'receiver_time_top10':receiver.most_common(10),'id_adjacent_descents':id_desc,'duration_counts':dict(dur_counts),'scan_elapsed_s':time.perf_counter()-start,'interpretation_note':'Structural chronology and schema audit only; no model trained. Exclude label-derived generator flags and identity/time metadata from predictors.'}
OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['rows','start_min_utc','start_max_utc','end_min_utc','end_max_utc','start_adjacent_descents','end_adjacent_descents','binary_label_counts','attack_type_counts','receiver_time_distinct','receiver_time_top10','id_adjacent_descents','scan_elapsed_s']},indent=2),flush=True)
print('wrote',OUT,flush=True)
