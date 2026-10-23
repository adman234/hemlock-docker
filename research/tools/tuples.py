import marshal, sys
def walk(c, path=""):
    for k in c.co_consts:
        if isinstance(k, tuple) and k and all(isinstance(x,str) for x in k):
            print(f"{path}{c.co_name}: {k}")
    for k in c.co_consts:
        if hasattr(k,'co_code'): walk(k, path+c.co_name+".")
walk(marshal.loads(open(sys.argv[1],'rb').read()))
