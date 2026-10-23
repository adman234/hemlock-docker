. C:\ns\refenv.ps1
$log='C:\ns\out\ref_runs.jsonl'
function Run($id,$file,$mode,$rq,$tgt,$con,$seed){
  $out="C:\ns\out\$id"; if(Test-Path $out){Remove-Item $out -Recurse -Force}
  $t=Get-Date
  $r=Ref "C:\ns\corpus\$file" $out $mode $rq $tgt $con $seed
  if(-not $r){ $r='{"ok":false,"error":"no RESULT line (process crashed)"}' }
  $o = $r | ConvertFrom-Json
  $o | Add-Member -NotePropertyName id -NotePropertyValue $id -Force
  $o | Add-Member -NotePropertyName wall_s -NotePropertyValue ((Get-Date)-$t).TotalSeconds -Force
  ($o | ConvertTo-Json -Compress) | Add-Content $log
}
# X-DIM
Run 'X-DIM-1' 'sq512.png' 1 0 dog cat
Run 'X-DIM-2' 'wide768.png' 1 0 dog cat
Run 'X-DIM-3' 'tall512x768.png' 1 0 dog cat
Run 'X-DIM-4' 'sq1024.png' 1 0 dog cat
Run 'X-DIM-5' 'big2000x1500.png' 1 0 dog cat
# X-FMT
Run 'X-FMT-1' 'fmt_rgba.png' 1 0 dog cat
Run 'X-FMT-2' 'fmt_gray.png' 1 0 dog cat
Run 'X-FMT-3' 'fmt_palette.png' 1 0 dog cat
Run 'X-FMT-4' 'wide768.jpg' 1 0 dog cat
Run 'X-FMT-5' 'fmt_exif6.jpg' 1 0 dog cat
Run 'X-FMT-6' 'fmt_still.gif' 1 0 dog cat
Run 'X-FMT-7' 'fmt_upper.PNG' 1 0 dog cat
# X-INT
Run 'X-INT-1' 'wide768.png' 0 0 dog cat
Run 'X-INT-3' 'wide768.png' 2 0 dog cat
# X-TGT
Run 'X-TGT-2' 'wide768.png' 1 0 car cat
Run 'X-TGT-3' 'wide768.png' 1 0 - cat
# X-REP
1..5 | % { Run "X-REP-$_" 'wide768.png' 1 0 dog cat }
# X-RQ
Run 'X-RQ-2' 'sq512.png' 1 1 dog cat
Run 'X-RQ-3' 'sq512.png' 1 2 dog cat
# X-BAD / X-SKIP
foreach($b in 'bad_zero.png','bad_trunc.png','bad_text.png','a-nightshade-intensity.png','a-protected-intensity-LOW-V1.png'){ Run ("X-BAD-"+$b) $b 1 0 dog cat }
Run 'X-RQ-4' 'sq512.png' 1 3 dog cat
'DONE' | Add-Content 'C:\ns\out\ref_matrix.done'
