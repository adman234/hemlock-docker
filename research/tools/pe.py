import pefile, sys
pe = pefile.PE(sys.argv[1])
print("machine", hex(pe.FILE_HEADER.Machine), "ts", pe.FILE_HEADER.TimeDateStamp, "subsys", pe.OPTIONAL_HEADER.Subsystem)
for fi in getattr(pe,'FileInfo',[]) or []:
    for e in fi:
        if hasattr(e,'StringTable'):
            for st in e.StringTable:
                for k,v in st.entries.items(): print(k.decode(), v.decode())
print("sections", [(s.Name.rstrip(b"\0").decode(), s.SizeOfRawData) for s in pe.sections])
print("signed", pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].Size)
