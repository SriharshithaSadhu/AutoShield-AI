"""Audit supplied notebooks without modifying or executing their full pipelines."""
import ast
import contextlib
import hashlib
import io
import itertools
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/base_objective_audit'


def exact_mask(w, alpha=.9, beta=.1):
    """Exact nonempty maximizer of additive weighted importance/count score."""
    coef=alpha*w/w.sum()-beta/len(w) if w.sum()>0 else np.full(len(w),-beta/len(w))
    mask=coef>0
    if not mask.any():
        mask[np.argmax(coef)]=True
    return mask


def main():
    OUT.mkdir(exist_ok=True)
    results=[]
    for nbname,dataname in [('MOO_AutoML_Dataset_1.ipynb','CICIDS2017_sample_km.csv'),
                            ('MOO_AutoML_Dataset_2.ipynb','IoTID20_sample_0.05.csv')]:
        path=ROOT/nbname
        nb=json.loads(path.read_text(encoding='utf-8'))
        code={i:''.join(c.get('source',[])) for i,c in enumerate(nb['cells']) if c['cell_type']=='code'}
        fs=next(i for i,s in code.items() if 'def get_original_feature_importance' in s)
        normal=next(i for i,s in code.items() if 'shapiro(df)' in s)
        objectives=[i for i,s in code.items() if 'def objective_function' in s and 'avg_confidence' in s]
        evidence='\n\n'.join(f'# Notebook cell {i} (zero-based)\n{code[i]}' for i in [normal,fs,*objectives])
        (OUT/(path.stem+'_extracted.py')).write_text(evidence,encoding='utf-8')
        df=pd.read_csv(ROOT/'data'/dataname)
        assert all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            stat,p=shapiro(df)
        finite_test=dict(statistic=float(stat),p=float(p),selected='zscore' if p>.05 else 'minmax',
                         flattened_n=int(df.size),includes_target='Label' in df,
                         warnings=[str(w.message) for w in ws])
        per_feature=[]
        for col in df.columns.drop('Label'):
            values=df[col].to_numpy(dtype=float)
            values=values[np.isfinite(values)]
            if len(values)<3 or np.ptp(values)==0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _,pv=shapiro(values)
            per_feature.append(dict(feature=col,p=float(pv)))
        # New diagnostic importance fit, NOT a faithful full-notebook reproduction.
        x=df.drop(columns='Label').replace([np.inf,-np.inf],np.nan)
        xt,_,yt,_=train_test_split(x,df.Label,test_size=.2,random_state=42,stratify=df.Label)
        model=lgb.LGBMClassifier(random_state=42,n_jobs=1,verbosity=-1).fit(xt,yt)
        w=model.feature_importances_.astype(float)
        # Execute original objective with a no-op estimator: its fitted outputs are unused.
        class Dummy:
            def fit(self,*args,**kwargs): return self
        class Namespace:
            LGBMClassifier=Dummy
        tree=ast.parse(code[fs])
        fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='objective_function')
        env=dict(np=np,lgb=Namespace)
        exec(compile(ast.Module(body=[fun],type_ignores=[]),'<notebook objective>','exec'),env)
        objective=env['objective_function']
        small=w[:12]
        frame=pd.DataFrame(np.zeros((2,12)))
        scores=[]
        for bits in itertools.product((False,True),repeat=12):
            if not any(bits): continue
            mask=np.array(bits)
            got=objective(mask.astype(float),frame,frame,[0,1],[0,1],small)[0]
            expected=.9*small[mask].sum()/small.sum()-.1*mask.mean() if small.sum() else -.1*mask.mean()
            assert np.isclose(got,expected)
            scores.append(got)
        em=exact_mask(small)
        assert np.isclose(objective(em.astype(float),frame,frame,[0,1],[0,1],small)[0],max(scores))
        full=exact_mask(w)
        # Original swarm mechanics and unchanged objective, with only unused fits stubbed out.
        swarm_env=dict(np=np,lgb=Namespace)
        exec(code[fs],swarm_env)
        swarm_env['get_original_feature_importance']=lambda *args: w
        f=pd.DataFrame(np.zeros((2,len(w))),columns=x.columns)
        exact_score=.9*w[full].sum()/w.sum()-.1*full.mean()
        runs=[]
        for seed in range(10):
            np.random.seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                mask,_,_=swarm_env['mopso'](f,f,[0,1],[0,1])
            score=.9*w[mask].sum()/w.sum()-.1*mask.mean()
            assert score<=exact_score+1e-12
            runs.append(dict(seed=seed,features=int(mask.sum()),score=float(score),gap=float(exact_score-score)))
        results.append(dict(notebook=nbname,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            cells=dict(normalization=normal,feature_selection=fs,model_objectives=objectives),
            normalization=finite_test,featurewise_nonconstant_count=len(per_feature),
            featurewise_accept_normality=sum(r['p']>.05 for r in per_feature),
            feature_selection=dict(importance_type=model.importance_type,features=len(w),
                importances=w.tolist(),feature_names=x.columns.tolist(),exact_selected=int(full.sum()),exact_score=float(exact_score),
                exhaustive_verified_masks=len(scores),swarm_objective_only_runs=runs,
                caveat='Fresh unnormalized 80/20 training fit, not the notebook resampled/normalized importance vector. Original objective and swarm executed with unused per-mask fits replaced by no-ops.'),
            objective_code_checks=dict(sigmoid_in_code=any('sigmoid' in s.lower() for s in code.values()),
                elapsed_seconds_includes_fit=all(s.index('start_time = time.time()')<s.index('clf.fit(')<s.index('elapsed_time = time.time() - start_time') for i in objectives for s in [code[i]]),
                sums_objectives=all('np.sum(fitness)' in code[i] for i in objectives),
                mean_max_probability=all('np.mean(np.max(proba, axis=1))' in code[i] for i in objectives))))
    out=dict(results=results,scope='Local supplied notebook audit, not attribution of experimental conduct to paper authors.')
    (OUT/'audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for r in results:
        print(r['notebook'],r['normalization'],r['objective_code_checks'])
        f=r['feature_selection']
        print('Exact selection',f['exact_selected'],'/',f['features'],'score',f['exact_score'],
              'swarm gaps',[round(x['gap'],6) for x in f['swarm_objective_only_runs']],
              'featurewise normality accepts',r['featurewise_accept_normality'],'/',r['featurewise_nonconstant_count'])


if __name__=='__main__': main()
