"""Shared deployment and evaluation utilities for the five-track development pilot."""
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, f1_score


def probabilities(bundle,x,calibrated=True):
    if 'feature_indices' in bundle:
        x=np.asarray(x)[:,bundle['feature_indices']]
    p=np.clip(bundle['pipeline'].predict_proba(x)[:,1],1e-12,1-1e-12)
    if calibrated:
        p=bundle['calibrator'].predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
    return p


def quality(y,p):
    pred=p>=.5
    confidence=np.maximum(p,1-p)
    correct=pred==y
    bins=np.minimum((confidence*15).astype(int),14)
    ece=sum((bins==i).mean()*abs(correct[bins==i].mean()-confidence[bins==i].mean())
            for i in range(15) if (bins==i).any())
    return dict(nll=float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
                brier=float(brier_score_loss(y,p)),ece15=float(ece),
                mean_confidence=float(confidence.mean()),macro_f1=float(f1_score(y,pred,average='macro')))


def confusion(y,p,t):
    pred=p>t
    nb,na=int((y==0).sum()),int((y==1).sum())
    fp,tp=int(((y==0)&pred).sum()),int(((y==1)&pred).sum())
    return dict(benign=nb,attack=na,fp=fp,tp=tp,fpr=fp/nb if nb else None,recall=tp/na if na else None)


def benign_threshold(y,p,cap):
    b=np.sort(p[y==0])
    return float(b[-1-int(np.floor(cap*len(b)))]) if len(b) else 1.


def conformal_thresholds(y,p,alpha=.1):
    scores=np.column_stack([p,1-p])
    qs=[]
    for c in (0,1):
        s=np.sort(scores[y==c,c])
        rank=int(np.ceil((len(s)+1)*(1-alpha)))
        qs.append(float(s[rank-1]) if 0<rank<=len(s) else 1.)
    return qs


def conformal_metrics(y,p,qs):
    sets=np.column_stack([p,1-p])<=np.array(qs)
    covered=sets[np.arange(len(y)),y.astype(int)]
    return dict(coverage=float(covered.mean()),mean_set_size=float(sets.sum(axis=1).mean()),
                empty_fraction=float((sets.sum(axis=1)==0).mean()),
                class_coverage={str(c):float(covered[y==c].mean()) if (y==c).any() else None for c in (0,1)})
