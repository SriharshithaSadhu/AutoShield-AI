"""Record bounded behavioral experiment before any development outcome scoring."""
from pathlib import Path
import json,datetime,hashlib
R=Path(__file__).resolve().parent;O=R/'results/behavioral_observation_2026-11-22'

def main():
    O.mkdir(exist_ok=True);p=O/'protocol.json'
    from behavioral_observation import FEATURE_NAMES
    record=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),
        primary_unit='Each observed completed exporter record. Host/window scores flag only the current record; no retrospective or future-flow attribution.',
        arrival_clock='API uses collector arrival_ms. Data lack export/receipt timestamps, so completion-time-sorted replay is an idealized ordered-collector scenario, not measured live timing.',
        visibility='One bidirectional collector receiving the provided records. Host-local rates are local to that collector, not verified complete host/network connections.',
        tie_rule='Only records with strictly earlier arrival timestamp affect current histories; same-time group is committed when arrival clock advances.',
        representation=FEATURE_NAMES,global_ablation=['log_global_record_rate10s','log_global_record_rate60s'],
        windows_ms=[10000,60000],ema_seconds=[10,60],host_cap=4096,per_host_contact_cap=64,
        cardinality='Exact min(distinct contacts in window,64) before any host/pending-host admission losses. Most-recent-unique-contact cache.',
        labels='No label enters observation state; labels used only for fitting, calibration, selection and evaluator accounting at their declared phases.',
        endpoint_policy='IPv4 values only ephemeral grouping/contact keys, never model features. Host churn/admission can erase history and is reported.',
        proxy_warning='Record count is not unique connection count; RST/SYN/no-reverse-packet flags are failure proxies, not verified connection outcomes. Not a TRW reproduction.',
        methods=['flow14','behavior24','quantile_tail_anomaly','hybrid38','behavior26_global_ablation'],
        seeds=[42,43,44],internal_fits=3,training_rows=200000,source_rows=dict(ton=150000,unsw=50000),
        classifier='Same100-tree15-leaf LightGBM and environment/class weights as strongest baseline. Flow fits reused.',
        calibration=dict(benign_per_network=20000,fpr_cap=.001,delta=.025),
        policies=['shared_pooled','shared_window','network_pooled','network_window'],
        full_replay=dict(source='ToN existing development tail',seed=42,policy='shared_pooled',
            original_flow_reference='Additional preserved shared_window baseline',batch_max=1000,batch_wait_ms=1000,
            episode='Evaluator-only(source,family) attack-record episodes,60s inactivity; label credit only current attack record. Censor episodes crossing evaluation start.',
            latency='Actual model calls plus measured observation cost; virtual batch waits and queueing derived from explicitly idealized arrival proxy. Export delay and real network delivery remain unmeasured.'),
        limits=dict(fpr=.001,minimum_recall=.5,model_bytes=1572864,p95_ms=5,resident_bytes=536870912),
        protected='UNSW final473085 rows filtered before state processing. Never scored, summarized by label or tuned.',
        novelty='No novelty claim. Conventional baselines and bounded causal observations; optimize only after demonstrated feasibility.',
        feature_code_sha256={name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['behavioral_observation.py','behavioral_data.py']})
    if p.exists():
        old=json.loads(p.read_text());assert old['feature_code_sha256']==record['feature_code_sha256']
    else:p.write_text(json.dumps(record,indent=2))
    print('Frozen behavioral protocol before model fitting/evaluation.')
if __name__=='__main__':main()
