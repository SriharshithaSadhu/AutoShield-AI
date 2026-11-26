from constrained_candidate_selection import select
c=dict(id=0,window_score=dict(worst_fpr=.0001,worst_fpr_upper=.0009),pooled_fpr=.0001,pooled_recall=.8,worst_recall=.6,minimum_family_recall=0,robust_recall_lower=0,mean_family_recall=.2,bytes=100000)
p={0:dict(native_p95_ms=1,peak_memory_bytes=100000000)}
assert select([c],p,'pooled')['selected']==0
assert select([c],p,'generalization_bound')['selected'] is None
assert select([c],{},'pooled')['selected'] is None
p[0]['peak_memory_bytes']=999999999
assert select([c],p,'pooled')['selected'] is None
print('PASS unknown resources, memory cap, recall feasibility and ordinary admission')
