"""Resume one official ZIP member by HTTP ranges; verify CRC and SHA-256 locally."""
import argparse
import hashlib
import http.client
import json
import shutil
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path
from time import sleep
from audit_corrected_cse_remote import Remote,URL

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/cse_corrected'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--member',default='Wednesday-28-02-2018.csv');args=parser.parse_args()
    if Path(args.member).name!=args.member:raise ValueError('basename only')
    OUT.mkdir(exist_ok=True)
    with (OUT / (args.member + '.lock')).open('a+b') as lock:
        lock.seek(0)
        if __import__('os').name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        download(args)


def download(args):
    remote=Remote()
    with zipfile.ZipFile(remote) as archive:info=archive.getinfo(args.member)
    if info.compress_type!=zipfile.ZIP_DEFLATED:raise ValueError('unsupported compression')
    remote.seek(info.header_offset);header=remote.read(30)
    assert header[:4]==b'PK\x03\x04'
    name_len,extra_len=struct.unpack('<HH',header[26:30])
    offset=info.header_offset+30+name_len+extra_len
    partial=OUT/(args.member+'.deflate.part');dest=OUT/args.member
    expected=dict(url=URL,member=args.member,etag=remote.headers.get('ETag'),offset=offset,
                  compressed_bytes=info.compress_size,uncompressed_bytes=info.file_size,crc32=info.CRC)
    metadata=OUT/(args.member+'.download.json')
    if metadata.exists():
        saved=json.loads(metadata.read_text())
        assert all(saved[k]==v for k,v in expected.items()),'Remote archive changed; refuse unsafe resume'
    else:metadata.write_text(json.dumps(expected,indent=2),encoding='utf-8')
    have=partial.stat().st_size if partial.exists() else 0
    assert have<=info.compress_size
    need=info.file_size+info.compress_size-have
    assert shutil.disk_usage(OUT).free>need+1_000_000_000,'Insufficient disk space'
    failures = 0
    while have < info.compress_size:
        stop = min(have + 16 * 1024 * 1024, info.compress_size)
        requested_range = f'{offset+have}-{offset+stop-1}'
        req = urllib.request.Request(URL, headers={'Range': f'bytes={requested_range}',
                                                   'If-Range': remote.headers['ETag']})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                content_range = response.headers.get('Content-Range', '').split('/')[0]
                if response.status != 206 or content_range != f'bytes {requested_range}':
                    raise ValueError(f'Unexpected range response: {response.status} {content_range}')
                with partial.open('ab') as stream:
                    while have < stop:
                        data = response.read(min(1024 * 1024, stop-have))
                        if not data:
                            raise IOError('Short range response')
                        stream.write(data)
                        have += len(data)
        except (OSError, http.client.HTTPException) as exc:
            failures += 1
            if failures >= 5:
                raise
            print(f'Retrying at byte {have}: {exc}', flush=True)
            sleep(2 ** failures)
        else:
            failures = 0
            print('downloaded', have, '/', info.compress_size, flush=True)
    temp=OUT/(args.member+'.tmp');decoder=zlib.decompressobj(-15);crc=0;sha=hashlib.sha256();n=0
    with partial.open('rb') as src,temp.open('wb') as dst:
        while block:=src.read(1024*1024):
            raw=decoder.decompress(block);dst.write(raw);crc=zlib.crc32(raw,crc);sha.update(raw);n+=len(raw)
        raw=decoder.flush();dst.write(raw);crc=zlib.crc32(raw,crc);sha.update(raw);n+=len(raw)
    assert decoder.eof and n==info.file_size and crc==info.CRC
    temp.replace(dest)
    expected.update(status='complete_crc_verified',sha256=sha.hexdigest(),bytes=n)
    metadata.write_text(json.dumps(expected,indent=2),encoding='utf-8')
    print(json.dumps(expected),flush=True)


if __name__=='__main__':main()
