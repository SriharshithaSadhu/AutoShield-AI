"""Verify every complete-role group collapses to the preserved category audit.

This independently checks the optimized counting pass, exact raw-key framing,
and protocol types needed for equivalence to the frozen numeric preprocessor.
"""
import sqlite3,zlib,json,math
from run_network_context_study import O

def main():
    b=O/'full_training_role_text_audit';result=json.loads((b/'results.json').read_text())
    previous=O/'full_training_text_audit';old=json.loads((previous/'results.json').read_text())
    conn=sqlite3.connect('file:'+str((b/'groups.sqlite').resolve())+'?mode=ro',uri=True)
    conn.execute('PRAGMA query_only=ON');conn.execute('ATTACH DATABASE ? AS previous',(str((previous/'groups.sqlite').resolve()),))
    protocols=set();checked=0
    def category_key(blob):
        nonlocal checked
        plain=zlib.decompress(blob);raw,signature=plain.rsplit(b'|context=',1)
        assert len(signature)==3 and raw.count(b'|')==45
        category,source,destination=signature
        assert 0<=category<=9 and source in [0,1] and destination in [0,1] and source+destination<=1
        assert (category>0)==(source+destination==1)
        proto=float(raw.split(b'|',1)[0]);assert math.isfinite(proto) and 0<=proto<=255 and proto==math.floor(proto)
        protocols.add(int(proto));checked+=1
        return zlib.compress(raw+b'|service='+str(category).encode('ascii'),1)
    conn.create_function('category_key',1,category_key,deterministic=True)
    groups,bad=conn.execute('''SELECT COUNT(*), SUM(CASE WHEN p.k IS NULL OR c.b!=p.benign OR c.a!=p.scanning THEN 1 ELSE 0 END)
        FROM (SELECT category_key(k) AS k, SUM(benign) AS b, SUM(scanning) AS a FROM main.groups GROUP BY category_key(k)) c
        LEFT JOIN previous.groups p ON c.k=p.k''').fetchone()
    assert groups==old['distinct_raw_text_vectors'] and bad==0,(groups,bad)
    actual=conn.execute('SELECT COUNT(*) FROM main.groups').fetchone()[0];assert actual==result['distinct_raw_text_vectors']
    conn.close()
    out=dict(complete_groups=actual,projected_category_groups=groups,mismatched_counts=bad,
        protocol_values=sorted(protocols),every_original_field_and_role_frame_valid=True,
        original_category_audit_preserved=True,
        statement='Every role-complete group projects exactly to category-only raw-text group counts, with no count mismatch. All protocol values are integral uint8 values; transport ports were validated by the frozen helper during counting.')
    (b/'key_refinement_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
