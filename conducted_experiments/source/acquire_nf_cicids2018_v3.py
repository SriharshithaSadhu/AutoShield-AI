"""Acquire public UQ NF-CICIDS2018-v3, CRC-check ZIP and verify publisher BagIt hashes."""
from pathlib import Path
import urllib.request,json,hashlib,zipfile,shutil,datetime
R=Path(__file__).resolve().parent;O=R/'data/nf_cicids2018_v3';UUID='4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22'
def main():
 O.mkdir(exist_ok=True);archive=O/'NF-CICIDS2018-v3_UQ.zip'
 with urllib.request.urlopen(f'https://api.rdm.uq.edu.au/datasets/{UUID}/download',timeout=30) as r:d=json.load(r)
 assert d['canDownload'] and d['accessType']=='OPEN'
 if not archive.exists():
  req=urllib.request.Request(d['url'],headers={'Range':'bytes=0-0'})
  with urllib.request.urlopen(req,timeout=60) as r:expected=int(r.headers.get('Content-Range','/0').split('/')[-1]) if r.status==206 else int(r.headers['Content-Length'])
  part=archive.with_suffix('.zip.part');offset=part.stat().st_size if part.exists() else 0
  req=urllib.request.Request(d['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
  with urllib.request.urlopen(req,timeout=120) as source:
   if offset and source.status!=206:raise RuntimeError('server did not honor resume; original partial preserved')
   with part.open('ab' if offset else 'wb') as sink:
    while True:
     block=source.read(4*1024**2)
     if not block:break
     sink.write(block)
  assert part.stat().st_size==expected,(part.stat().st_size,expected)
  part.replace(archive)
 with zipfile.ZipFile(archive) as z:
  bad=z.testzip();assert bad is None,bad
  names=z.namelist();print('ZIP members',names,flush=True)
  checksums={}
  for name in names:
   if name.endswith('manifest-sha1.txt'):
    for line in z.read(name).decode().splitlines():
     digest,member=line.split(None,1);checksums[Path(member.strip()).name]=digest
  extracted={}
  for name in names:
   if name.lower().endswith('.csv'):
    output=O/Path(name).name
    if not output.exists():
     temporary=output.with_suffix(output.suffix+'.extracting')
     with z.open(name) as source,temporary.open('wb') as sink:shutil.copyfileobj(source,sink,4*1024**2)
     temporary.replace(output)
    sha=hashlib.sha1()
    with output.open('rb') as f:
     for block in iter(lambda:f.read(4*1024**2),b''):sha.update(block)
    actual=sha.hexdigest();expected=checksums.get(output.name)
    if expected is not None:assert actual==expected,(output.name,actual,expected)
    extracted[output.name]=dict(bytes=output.stat().st_size,sha1=actual,publisher_sha1=expected,verified=expected is not None)
  # Preserve supplied metadata/rights text without extracting arbitrary paths.
  for name in names:
   if name.lower().endswith(('.txt','.json')) and not name.endswith('/'):
    (O/('archive_'+Path(name).name)).write_bytes(z.read(name))
 digest=hashlib.sha256()
 with archive.open('rb') as source:
  for block in iter(lambda:source.read(4*1024**2),b''):digest.update(block)
 sha256=digest.hexdigest()
 report=dict(dataset='NF-CICIDS2018-v3',record=f'https://rdm.uq.edu.au/files/{UUID}',access='OPEN',archive_bytes=archive.stat().st_size,archive_sha256=sha256,zip_crc='pass',payloads=extracted,downloaded_at=datetime.datetime.now().astimezone().isoformat(),scope='Official UQ public archive; locally calculated archive hash, publisher BagIt payload hashes where present. Research use; retain and audit supplied rights metadata.')
 (O/'download_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
