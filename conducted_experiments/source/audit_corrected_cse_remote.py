"""Inspect official corrected ZIP metadata and CSV prefixes with bounded HTTP ranges."""
import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL='https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip'
OUT=Path(__file__).resolve().parent/'results/corrected_cse_gate'


class Remote(io.RawIOBase):
    def __init__(self):
        with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=45) as r:
            self.headers=dict(r.headers);self.size=int(r.headers['Content-Length'])
        self.pos=0;self.downloaded=0;self.requests=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if n>2_000_000 or self.downloaded+n>8_000_000:raise ValueError('metadata audit transfer bound exceeded')
        req=urllib.request.Request(URL,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'})
        with urllib.request.urlopen(req,timeout=45) as r:
            if r.status!=206:raise RuntimeError('Server ignored range; refused full archive transfer')
            data=r.read(n+1)
            if len(data)!=n:raise RuntimeError('Unexpected range response length')
        self.downloaded+=len(data);self.requests+=1;self.pos+=len(data)
        return data


def main():
    OUT.mkdir(exist_ok=True)
    remote=Remote();catalog=[];samples=[]
    with zipfile.ZipFile(remote) as z:
        for info in z.infolist():
            catalog.append(dict(name=info.filename,compressed_bytes=info.compress_size,bytes=info.file_size,crc32=info.CRC))
            if any(date in info.filename for date in ('28-02-2018','01-03-2018','02-03-2018')) and info.filename.endswith('.csv'):
                with z.open(info) as f:
                    prefix=f.read(16000).decode('utf-8-sig',errors='replace')
                rows=list(csv.reader(prefix.splitlines()))
                samples.append(dict(name=info.filename,header=rows[0],sample_rows=rows[1:4],
                                    caveat='Prefix only, no full-file CRC verification or temporal/label audit'))
                print('inspected',info.filename,flush=True)
    out=dict(url=URL,response_headers=remote.headers,archive_bytes=remote.size,bytes_transferred=remote.downloaded,
             http_requests=remote.requests,catalog=catalog,samples=samples,
             status='metadata/prefix gate only; not acquired, validated or approved for replay')
    (OUT/'remote_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
