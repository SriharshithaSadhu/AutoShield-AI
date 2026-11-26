"""Create a local portable benchmark bundle; no publication or device claim."""
from pathlib import Path
import datetime,hashlib,json,shutil,zipfile
import pandas as pd
from behavioral_runtime import RAW_FIELDS
from behavioral_data import META_COLUMNS
R=Path(__file__).resolve().parent;O=R/'results/behavioral_observation_2026-11-22';B=R/'deployment/raspberry_pi_behavioral'
def main():
    B.mkdir(exist_ok=True);models={}
    for name in ['behavioral_observation.py','behavioral_models.py','behavioral_runtime.py','multienvironment_features.py']:shutil.copy2(R/name,B/name)
    for c in json.loads((O/'selections.json').read_text()):
        if c['seed']!=42:continue
        s=c['selected']['shared_pooled'];name=c['method']+'.joblib';shutil.copy2(R/s['artifact'],B/name)
        models[c['method']]=dict(file=name,threshold=s['thresholds']['shared'],source_artifact=s['artifact'],status='experimental_not_deployment_approved')
    fields=list(dict.fromkeys(META_COLUMNS+list(RAW_FIELDS)))
    pd.read_csv(R/'data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv',usecols=fields,nrows=6000).to_csv(B/'trace.csv',index=False)
    (B/'requirements.txt').write_text('numpy==1.26.4\nlightgbm==4.5.0\nscikit-learn==1.3.2\njoblib==1.4.2\npsutil==7.2.2\nthreadpoolctl\n')
    (B/'README.md').write_text('AutoShield experimental behavioral runtime benchmark.\n\nUse Python3.10 or3.11 in a separate environment. Install requirements, then run `python benchmark_device.py`.\n\nThe output records actual hardware identity; a desktop run is not a Raspberry Pi result. The trace contains6000 unlabeled genuine NF-ToN-IoT-v3 ordered-prefix records. Source dataset reuse terms: Permitted Re-Use with Commercial Use Restriction (see original publisher). This local bundle is for research benchmarking, not a dataset redistribution release.\n\nNo method is deployment approved. Results measure runtime service only, excluding exporter delay, collector/network IO, batch wait and energy. Batch1 and batch1000 timings use different units.\n')
    names=['benchmark_device.py','requirements.txt','README.md','trace.csv',*['behavioral_observation.py','behavioral_models.py','behavioral_runtime.py','multienvironment_features.py'],*[m['file'] for m in models.values()]]
    manifest=dict(built_at=datetime.datetime.now().astimezone().isoformat(),models=models,trace='6000 genuine ordered-prefix ToN flows; labels excluded; not a new validation set.',
        sha256={name:hashlib.sha256((B/name).read_bytes()).hexdigest() for name in names})
    (B/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(O/'AutoShield_Behavioral_Device_Benchmark.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names+['manifest.json']:z.write(B/name,name)
    print('BUILT local device benchmark',len(names)+1,'files',flush=True)
if __name__=='__main__':main()
