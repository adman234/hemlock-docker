$env:USERPROFILE='C:\ns\home'; $env:HOME='C:\ns\home'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONWARNINGS='ignore'
$src='C:\Users\Adam\Documents\GitHub\hemlock-docker\nightshade-reimplementation\src'
function Re($img,$out,$mode,$rq,$tgt,$seed){ $a=@('-I','C:\Users\Adam\Documents\GitHub\hemlock-docker\research\tools\reimpl_run.py',$src,$img,$out,$mode,$rq,$tgt); if($seed -ne $null){$a+=$seed}; & C:\ns\refenv\Scripts\python.exe @a 2>$null | Select-String '^RESULT' | % { $_.Line.Substring(7) } }
