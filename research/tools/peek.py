import pickle, io, numpy as np
class U(pickle.Unpickler):
    def find_class(self, m, n):
        if (m,n) in {("numpy.core.multiarray","_reconstruct"),("numpy","ndarray"),("numpy","dtype"),("numpy._core.multiarray","_reconstruct")}:
            return super().find_class(m,n)
        raise pickle.UnpicklingError(f"blocked {m}.{n}")
d = U(open(r"C:\ns\work\full_asset3.dec.pkl","rb")).load()
print(type(d), list(d.keys()) if isinstance(d,dict) else len(d))
for k,v in d.items():
    if isinstance(v,np.ndarray): print(k,"ndarray",v.shape,v.dtype, "norm0",float(np.linalg.norm(v[0])))
    elif isinstance(v,(list,tuple)): print(k,type(v).__name__,len(v),v[:12])
    elif isinstance(v,dict): print(k,"dict",len(v),list(v.items())[:8])
    else: print(k,type(v))
