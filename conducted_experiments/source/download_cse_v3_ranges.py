"""Resumable four-worker HTTP Range transfer of official public UQ archive."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import urllib.request,json,time
O=Path(__file__).resolve().parent/'data/nf_cicids2018_v3';A=O/'NF-CICIDS2018-v3_UQ.zip';S=O/'range_segments';S.mkdir(exist_ok=True)
with urllib.request.urlopen('https://api.rdm.uq.edu.au/datasets/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22/download',timeout=30) as r:d=json.load(r)
assert d['canDownload'] and d['accessType']=='OPEN'
url=d['url']
with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=30) as r:assert r.status==206;total=int(r.headers['Content-Range'].split('/')[-1])
block=16*1024**2

def fetch(start):
 end=min(total,start+block)-1;p=S/f'{start:012d}.part'
 for attempt in range(3):
  have=p.stat().st_size if p.exists() else 0
  if have==end-start+1:return p
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start+have}-{end}'})
   with urllib.request.urlopen(req,timeout=30) as r:
    assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start+have}-')
    with p.open('ab') as sink:
     deadline=time.monotonic()+120
     while True:
      b=r.read(256*1024)
      if not b:break
      sink.write(b)
      if time.monotonic()>deadline:raise TimeoutError('segment deadline')
   assert p.stat().st_size==end-start+1
   return p
  except Exception:
   if attempt==2:raise
 print('unreachable')
with ThreadPoolExecutor(max_workers=4) as pool:
 futures=[pool.submit(fetch,s) for s in range(0,total,block)]
 for i,f in enumerate(as_completed(futures),1):f.result();print('verified-length segments',i,'of',len(futures),flush=True)
combined=A.with_suffix('.assembling')
with combined.open('wb') as sink:
 for start in range(0,total,block):
  with (S/f'{start:012d}.part').open('rb') as source:
   for b in iter(lambda:source.read(1024**2),b''):sink.write(b)
assert combined.stat().st_size==total;combined.replace(A);print('assembled official archive',total,flush=True)
