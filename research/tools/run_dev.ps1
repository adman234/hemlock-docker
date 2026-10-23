. C:\ns\refenv.ps1
. C:\ns\reenv.ps1
$log='C:\ns\out\dev_runs.jsonl'
function Log($id,$r,$t){ if(-not $r){ $r='{"ok":false,"error":"no RESULT line"}' }; $o=$r|ConvertFrom-Json; $o|Add-Member -NotePropertyName id -NotePropertyValue $id -Force; $o|Add-Member -NotePropertyName wall_s -NotePropertyValue ((Get-Date)-$t).TotalSeconds -Force; ($o|ConvertTo-Json -Compress)|Add-Content $log }
$env:CUDA_VISIBLE_DEVICES='-1'
$t=Get-Date; Remove-Item C:\ns\out\re_D-CPU-S -Recurse -Force -ErrorAction SilentlyContinue; Log 'D-CPU-S' (Re C:\ns\corpus\sq512.png C:\ns\out\re_D-CPU-S 1 0 dog 123) $t
$t=Get-Date; Remove-Item C:\ns\out\D-CPU-REF -Recurse -Force -ErrorAction SilentlyContinue; Log 'D-CPU-REF' (Ref C:\ns\corpus\sq512.png C:\ns\out\D-CPU-REF 1 0 dog cat) $t
'DONE' | Add-Content C:\ns\out\dev.done

