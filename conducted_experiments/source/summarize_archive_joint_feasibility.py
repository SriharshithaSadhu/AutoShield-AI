"""Render diagnostic evidence without choosing a method from evaluation labels."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT/'results/archive_joint_feasibility.json').read_text())
    lines = ['# Joint false-positive / recall feasibility pilot', '',
             'Exploratory CSE development results. No online switching or novel-method advantage established.', '',
             'Stages: ' + json.dumps(data['protocol']), '',
             'FPR and recall below are pooled over every sampled evaluation window. Joint passes require >=100 benign and >=100 attack flows per window.', '',
             '| Seed | Model | FPR cap | Evaluation FPR | Evaluation recall | Joint passes (recall >=50%) | Fit CPU s | Model KiB |',
             '|---|---|---|---|---|---|---|---|']
    for result in data['results']:
        for exp in result['experiments']:
            if exp['recall_floor'] != .5:
                continue
            for model, stats in zip(result['models'], exp['evaluation']):
                lines.append(f"| {result['seed']} | {model['name']} | {exp['cap']:.1%} | {stats['fpr']:.4%} | {stats['recall']:.3%} | {stats['joint_pass']}/{stats['supported']} | {model['fit_cpu_s']:.2f} | {model['bytes']/1024:.1f} |")
    lines += ['', '## Prefix-selected choices', '', '| Seed | Cap | Recall floor | Selected reserve | Prefix passes | Later passes |', '|---|---|---|---|---|---|']
    for result in data['results']:
        for e in result['experiments']:
            i = e['selected']
            lines.append(f"| {result['seed']} | {e['cap']:.1%} | {e['recall_floor']:.0%} | {result['models'][i]['name'] if i is not None else 'none'} | {e['selection'][i]['joint_pass'] if i is not None else '-'} | {e['evaluation'][i]['joint_pass'] if i is not None else '-'} |")
    lines += ['', '## Hindsight diagnostic, not a deployable policy', '',
              'The following counts ask whether any candidate could meet both constraints in each supported later window. This uses evaluation labels and is only an upper bound on routing potential within this fixed candidate/threshold set.', '',
              '| Seed | Cap | Recall floor | Any source | Any target archive | Fixed target baseline | Any model | Supported windows |', '|---|---|---|---|---|---|---|---|']
    for result in data['results']:
        for e in result['experiments']:
            def passes(r):
                return min(r['benign'], r['attack']) >= 100 and r['fp']/r['benign'] <= e['cap'] and r['tp']/r['attack'] >= e['recall_floor']
            counts = [sum(any(passes(e['records'][i][j]) for i in ids) for j in range(len(e['records'][0]))) for ids in (range(4),range(4,8),[8],range(9))]
            lines.append(f"| {result['seed']} | {e['cap']:.1%} | {e['recall_floor']:.0%} | " + ' | '.join(map(str,counts)) + f" | {e['evaluation'][0]['supported']} |")
    lines += ['', '## Limitations', '', *['- '+s for s in data['limitations']], '',
              '- Target-model fits use the same rows within each seed; source fits use source data. No matched-label comparison across domains.',
              '- Cache excludes the initial source replay prefix; this is not all ten raw capture files.',
              '- Low class counts and temporal dependence limit statistical interpretation; no finite-sample guarantee is asserted.',
              '- A source-archive model refitted on target labels is not a reproduction of the full base-paper AutoML search on target data.', '']
    (ROOT/'results/ARCHIVE_JOINT_FEASIBILITY.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
