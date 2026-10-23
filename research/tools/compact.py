import re,sys
out=[]
for l in open(sys.argv[1],encoding='utf8'):
    l=l.rstrip()
    if not l.strip(): continue
    if re.search(r'\b(PRECALL|PUSH_NULL|RESUME|CACHE|NOP|EXTENDED_ARG|COPY_FREE_VARS|MAKE_CELL)\b',l): continue
    m=re.match(r'\s*(\d+)?\s+(>>)?\s*\d+ (\w+)\s*(.*)',l)
    if m:
        ln,_,op,arg=m.groups()
        arg=re.sub(r'^\d+ \(','(',arg)
        arg=re.sub(r'<code object (\w+|<\w+>) at 0x[0-9A-F]+, file "[^"]+", line (\d+)>',r'<code \1@\2>',arg)
        out.append(('L%s '%ln if ln else '    ')+op+' '+arg)
    else:
        out.append(re.sub(r' at 0x[0-9A-F]+, file "[^"]+"','',l))
open(sys.argv[2],'w',encoding='utf8').write('\n'.join(out))
