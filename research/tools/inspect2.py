import pickletools, zipfile, sys, collections
def globals_of(data, label):
    mods = collections.Counter(); last=[]
    for op,arg,pos in pickletools.genops(data):
        if op.name in ("GLOBAL","INST"): mods[arg.replace(" ",".")]+=1
        elif op.name=="STACK_GLOBAL" and len(last)>=2: mods[f"{last[-2]}.{last[-1]}"]+=1
        if op.name in ("SHORT_BINUNICODE","BINUNICODE","UNICODE","BINUNICODE8"): last.append(arg)
        last=last[-4:]
    print(f"== {label}: {len(data):,} bytes")
    for k,v in mods.most_common(60): print("   ",v,k)
p=sys.argv[1]
if p.endswith("clip_model.p"):
    z=zipfile.ZipFile(p); n=[x for x in z.namelist() if x.endswith("data.pkl")][0]
    print(n, len(z.namelist())); globals_of(z.read(n), p)
else:
    from cryptography.fernet import Fernet
    import re, subprocess
    key=sys.argv[2].encode()
    data=Fernet(key).decrypt(open(p,"rb").read())
    open(sys.argv[3],"wb").write(data)
    globals_of(data,p+" (decrypted)")
