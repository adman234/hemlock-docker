import sys, zlib, marshal, os
from PyInstaller.archive.readers import CArchiveReader
exe, out = sys.argv[1], sys.argv[2]
ar = CArchiveReader(exe)
os.makedirs(out, exist_ok=True)
for name, (off, dlen, ulen, flag, typ) in ar.toc.items():
    print(typ, name, ulen)
    data = ar.extract(name)
    if typ == 'z' or typ=='Z':
        pass
    p = os.path.join(out, name.replace('/', os.sep))
    os.makedirs(os.path.dirname(p) or out, exist_ok=True)
    with open(p, 'wb') as f: f.write(data if isinstance(data,(bytes,bytearray)) else b'')
