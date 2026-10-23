. C:\ns\reenv.ps1
$log='C:\ns\out\re_runs.jsonl'
function Run($id,$file,$mode,$rq,$tgt,$seed){
  $out="C:\ns\out\re_$id"; if(Test-Path $out){Remove-Item $out -Recurse -Force}
  $t=Get-Date
  $r=Re "C:\ns\corpus\$file" $out $mode $rq $tgt $seed
  if(-not $r){ $r='{"ok":false,"error":"no RESULT line (process crashed)"}' }
  $o = $r | ConvertFrom-Json
  $o | Add-Member -NotePropertyName id -NotePropertyValue $id -Force
  $o | Add-Member -NotePropertyName wall_s -NotePropertyValue ((Get-Date)-$t).TotalSeconds -Force
  ($o | ConvertTo-Json -Compress) | Add-Content $log
}
Run 'X-DIM-1' 'sq512.png' 1 0 dog
Run 'X-DIM-2' 'wide768.png' 1 0 dog
Run 'X-DIM-3' 'tall512x768.png' 1 0 dog
Run 'X-DIM-4' 'sq1024.png' 1 0 dog
Run 'X-DIM-5' 'big2000x1500.png' 1 0 dog
Run 'X-FMT-1' 'fmt_rgba.png' 1 0 dog
Run 'X-FMT-2' 'fmt_gray.png' 1 0 dog
Run 'X-FMT-3' 'fmt_palette.png' 1 0 dog
Run 'X-FMT-4' 'wide768.jpg' 1 0 dog
Run 'X-FMT-5' 'fmt_exif6.jpg' 1 0 dog
Run 'X-FMT-6' 'fmt_still.gif' 1 0 dog
Run 'X-FMT-7' 'fmt_upper.PNG' 1 0 dog
Run 'X-INT-1' 'wide768.png' 0 0 dog
Run 'X-INT-3' 'wide768.png' 2 0 dog
Run 'X-TGT-2' 'wide768.png' 1 0 car
1..5 | % { Run "X-REP-$_" 'wide768.png' 1 0 dog }
Run 'X-RQ-2' 'sq512.png' 1 1 dog
Run 'X-RQ-3' 'sq512.png' 1 2 dog
Run 'X-RQ-4' 'sq512.png' 1 3 dog
# seeded determinism of the reimplementation (opt-in feature)
Run 'S-SEED-1' 'wide768.png' 1 0 dog 123
Run 'S-SEED-2' 'wide768.png' 1 0 dog 123
'DONE' | Add-Content 'C:\ns\out\re_matrix.done'
