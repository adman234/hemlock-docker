import sys, os, marshal
from PyInstaller.archive.readers import ZlibArchiveReader
ar = ZlibArchiveReader(sys.argv[1])
out = sys.argv[2]
names = [n for n in ar.toc]
print(len(names))
keep = [n for n in names if n.split('.')[0] in ('nightshade','utils','downloader','app','ImageAnalyzer','glaze','nightshade_resources','clip','lpips') or not n.split('.')[0] in ('torch','transformers','scipy','sklearn','numpy','PIL','PyQt6','diffusers','huggingface_hub','nltk','tokenizers','safetensors','torchvision','yaml','requests','urllib3','cryptography','regex','psutil','tqdm','packaging','markupsafe','filelock','click','certifi','wcwidth','ftfy','idna','charset_normalizer','encodings','importlib','email','http','json','xml','unittest','distutils','pkg_resources','setuptools','_distutils_hack','asyncio','concurrent','multiprocessing','logging','ctypes','collections','urllib','test','html','sqlite3','tkinter','pydoc_data','lib2to3','zipfile','pathlib','typing_extensions','zipp','jaraco','more_itertools','platformdirs','sympy','mpmath','networkx','jinja2','fsspec','joblib','threadpoolctl','six','dateutil','pytz','attr','colorama','win32','pywin32_system32','pythoncom','pywintypes','curses','dbm','ensurepip','idlelib','turtledemo','venv','wsgiref','xmlrpc','zoneinfo')]
print(keep)
os.makedirs(out, exist_ok=True)
for n in keep:
    typ, code = ar.toc[n] if False else (None,None)
    c = ar.extract(n)
    with open(os.path.join(out, n+'.marshal'),'wb') as f: f.write(marshal.dumps(c))
