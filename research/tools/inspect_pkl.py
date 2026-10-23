import pickletools, zipfile, sys, io, collections
def globals_of(data, label):
    mods = collections.Counter(); strs=[]
    last=[]
    for op,arg,pos in pickletools.genops(data):
        if op.name in ("GLOBAL","INST"): mods[arg.replace(" ",".")]+=1
        elif op.name=="STACK_GLOBAL" and len(last)>=2: mods[f"{last[-2]}.{last[-1]}"]+=1
        if op.name in ("SHORT_BINUNICODE","BINUNICODE","UNICODE","BINUNICODE8"): last.append(arg)
        last=last[-4:]
    print(f"== {label}: {len(data):,} bytes, {sum(mods.values())} global refs")
    for k,v in mods.most_common(40): print("   ",v,k)
p=sys.argv[1]
if p.endswith(".pt"):
    z=zipfile.ZipFile(p); n=[x for x in z.namelist() if x.endswith("data.pkl")][0]
    print(n, [x for x in z.namelist()][:6], len(z.namelist()))
    globals_of(z.read(n), p)
else:
    globals_of(open(p,"rb").read(), p)
