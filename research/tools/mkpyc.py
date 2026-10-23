import marshal, importlib.util, sys, os
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for m in ("nightshade","opt","utils","downloader"):
    code = open(os.path.join(src, m + ".marshal"), "rb").read()
    open(os.path.join(out, m + ".pyc"), "wb").write(importlib.util.MAGIC_NUMBER + b"\0"*12 + code)
print(os.listdir(out))
