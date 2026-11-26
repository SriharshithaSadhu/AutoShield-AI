"""Render the completed full-capture audit without replacing prior provenance notes."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/corrected_cse_gate'
def main():
    d=json.loads((OUT/'Wednesday-28-02-2018.csv.full_audit.json').read_text());c=d['counts']
    assert sum(d['labels'].values())==c['rows']
    assert c['primary_attack']+c['primary_benign']==c['rows']
    bins=json.loads((OUT/'Wednesday-28-02-2018.csv.minute_bins.json').read_text())
    assert sum(sum(v) for v in bins.values())==c['rows']-c['invalid_start']-c['negative_duration']-c['nonfinite_duration']
    assert sum(v[1] for v in bins.values())==c['primary_attack']
    lines=['# Complete corrected-CSE capture audit','',
        f"Capture: {d['member']}. Full CRC-verified CSV: {d['provenance']['bytes']:,} bytes; {c['rows']:,} rows.",
        f"SHA-256: `{d['provenance']['sha256']}`.", '',
        '| Check | Result |','|---|---|',
        f"| Primary benign / attack | {c['primary_benign']:,} / {c['primary_attack']:,} |",
        f"| Attempted flows mapped benign | {c['attempted_flows']:,} |",
        f"| Invalid timestamps / negative duration / nonfinite duration | {c['invalid_start']} / {c['negative_duration']} / {c['nonfinite_duration']} |",
        f"| Zero-duration flows | {c['zero_duration']:,} |",
        f"| Completion-time reversals in file | {c['completion_descents_in_file']:,} |",
        f"| Minute bins / both-class bins / bins with >=100 each class | {d['minute_bins']} / {d['both_class_minute_bins']} / {d['supported_minute_bins']} |",
        f"| Completion-time span | {pd.Timestamp(d['completion_min_ns'])} to {pd.Timestamp(d['completion_max_ns'])} |"]
    for key,val in d['rate_invariants'].items():
        lines.append(f"| {key} packet/rate/duration consistency | {val['matches']:,} / {val['eligible']:,} |")
    lines+=['','Raw label counts: `'+json.dumps(d['labels'])+'`.', '',
        'The source file is not assumed chronological. Replay must sort by completion time with deterministic tie handling before simulating release of chunk labels. Completion time is an event-time surrogate; exporter-arrival timing remains unknown.',
        'This is one corrected development capture, not independent confirmation. All 77 feature semantics, duplicate/conflicting vectors, source-extractor compatibility, and chronological cache construction remain open. No model or adaptation metrics were computed on this capture in this audit.',
        'See EXTRACTOR_SEMANTICS.md for implementation evidence and REPORT.md for official provenance/attribution. Row counts, label totals and minute-bin totals reconcile.']
    (OUT/'FULL_CAPTURE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__':main()
