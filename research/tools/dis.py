import marshal, dis, sys, io
c=marshal.loads(open(sys.argv[1],'rb').read())
dis.dis(c)
