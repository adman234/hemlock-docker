"""HTML, CSS and JS for the results: before/after cards, the full-resolution viewer, progress bar, status pill.

Gradio sets HTML component content with innerHTML, which never runs <script> tags, so the behaviour lives in one
script injected through gr.Blocks(head=...) and uses event delegation: it keeps working for cards added later.
"""
import html

e = html.escape

CSS = """
.gradio-container{max-width:760px !important;margin:0 auto !important;padding-left:12px !important;padding-right:12px !important}
.gradio-container input,.gradio-container textarea{font-size:16px !important}  /* stops iOS zooming on focus */
button.lg,button.primary{min-height:52px;font-size:1.05rem}
#hero h1{margin:.2em 0 .1em;font-size:1.9rem}
#hero p{margin:.2em 0;opacity:.8}
.note{border:1px solid var(--border-color-primary,#8884);border-left:4px solid #10b981;border-radius:10px;padding:10px 14px;margin:10px 0;background:var(--background-fill-secondary,#8881);font-size:.92rem;line-height:1.45}
.about{font-size:.92rem;line-height:1.5}.about p,.about ul{margin:.5em 0}.about li{margin:.35em 0}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.85rem;margin:6px 6px 6px 0}
.pill.ok{background:#1f9d5533;color:#1f9d55}.pill.wait{background:#d9a40033;color:#b58900}.pill.bad{background:#d6336c33;color:#d6336c}
.pill.idle{background:#8883;color:inherit}
.prog{border:1px solid var(--border-color-primary,#8884);border-radius:12px;padding:12px;background:var(--background-fill-secondary,#8881)}
.prog .msg{font-weight:600;margin-bottom:8px}.prog .sub{opacity:.7;font-size:.85rem;margin-top:6px}
.prog .bar{height:10px;border-radius:999px;background:#8883;overflow:hidden}.prog .bar i{display:block;height:100%;background:#10b981;transition:width .4s}
.cmp-list{display:flex;flex-direction:column;gap:20px}
.cmp{border:1px solid var(--border-color-primary,#8884);border-radius:14px;padding:10px;background:var(--background-fill-secondary,#8881)}
.cmp-head,.cmp-foot{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.cmp-name{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:55%}
.cmp-btns{display:flex;gap:8px}
.cmp-dl,.cmp-mode,.cmp-full,.copy{padding:8px 14px;min-height:40px;border-radius:10px;border:1px solid var(--border-color-primary,#8886);
  background:var(--button-secondary-background-fill,#8882);color:inherit;font-size:.9rem;text-decoration:none;cursor:pointer;
  display:inline-flex;align-items:center}
.copy{min-height:32px;padding:3px 10px;margin-left:8px;font-size:.8rem}
.stage{position:relative;margin:10px 0;border-radius:10px;overflow:hidden;aspect-ratio:var(--ar);width:100%;max-height:75vh;
  background:#0003;touch-action:pan-y;user-select:none;-webkit-user-select:none;-webkit-touch-callout:none;cursor:pointer;outline:none}
.stage:focus-visible{box-shadow:0 0 0 3px #10b98188}
.stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;-webkit-user-drag:none}
.img-o{opacity:0}
.cmp[data-mode=hold] .stage.holding .img-o{opacity:1}
.tag{position:absolute;left:8px;top:8px;padding:3px 10px;border-radius:999px;font-size:.78rem;color:#fff;background:#000a;pointer-events:none}
.tag-o{opacity:0}
.cmp[data-mode=hold] .stage.holding .tag-o{opacity:1}.cmp[data-mode=hold] .stage.holding .tag-m{opacity:0}
.range,.divider{display:none}
.cmp[data-mode=slide] .img-o{opacity:1;clip-path:inset(0 calc(100% - var(--pos,50%)) 0 0)}
.cmp[data-mode=slide] .tag-o{opacity:1;left:8px}.cmp[data-mode=slide] .tag-m{left:auto;right:8px}
.cmp[data-mode=slide] .stage{cursor:ew-resize}
.cmp[data-mode=slide] .range{display:block;position:absolute;inset:0;width:100%;height:100%;opacity:0;margin:0;cursor:ew-resize;z-index:3}
.cmp[data-mode=slide] .divider{display:block;position:absolute;top:0;bottom:0;left:var(--pos,50%);width:3px;margin-left:-1.5px;background:#fff;
  box-shadow:0 0 6px #0008;pointer-events:none;z-index:2}
.hint{opacity:.7;font-size:.85rem}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip,.stat{display:inline-flex;align-items:center;padding:3px 10px;border-radius:999px;font-size:.8rem;background:#8882}
.chip b{margin-left:.3em}
.stat{opacity:.8}
/* full-resolution viewer */
#hm-full{position:fixed;inset:0;z-index:100000;background:#000e;display:flex;flex-direction:column}
#hm-full[hidden]{display:none}
.fs-bar{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#111;color:#fff;flex-wrap:wrap}
.fs-title{flex:1;min-width:140px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fs-bar button{padding:8px 14px;min-height:40px;border-radius:10px;border:1px solid #fff4;background:#fff1;color:#fff;font-size:.9rem;cursor:pointer}
.fs-bar button.on{background:#10b981;border-color:#10b981}
.fs-body{flex:1;overflow:auto;display:grid;place-items:center;padding:8px}
.fs-stage{position:relative;display:inline-block;line-height:0;user-select:none;-webkit-user-select:none;cursor:pointer}
.fs-stage img{display:block;-webkit-user-drag:none}
.fs-stage[data-fit=fit] .fs-m{max-width:calc(100vw - 16px);max-height:calc(100vh - 80px)}
.fs-stage[data-fit=full] .fs-m{max-width:none;max-height:none}
.fs-stage .fs-o{position:absolute;inset:0;width:100%;height:100%;opacity:0;pointer-events:none}
.fs-stage[data-mode=hold].holding .fs-o{opacity:1}
.fs-stage[data-mode=slide] .fs-o{opacity:1;clip-path:inset(0 calc(100% - var(--fpos,50%)) 0 0)}
.fs-range,.fs-div{display:none}
.fs-stage[data-mode=slide] .fs-range{display:block;position:absolute;inset:0;width:100%;height:100%;opacity:0;margin:0;cursor:ew-resize;z-index:3}
.fs-stage[data-mode=slide] .fs-div{display:block;position:absolute;top:0;bottom:0;left:var(--fpos,50%);width:3px;margin-left:-1.5px;background:#fff;box-shadow:0 0 6px #0008;pointer-events:none;z-index:2}
.fs-note{color:#fff9;font-size:.8rem;padding:4px 12px;background:#111}
/* desktop: wider layout, larger type, bigger previews */
@media (min-width:1000px){
  html{font-size:17px}
  .gradio-container{max-width:min(1500px,95vw) !important}
  #hero h1{font-size:2.4rem}
  .stage{max-height:82vh}
  .cmp{padding:16px}
}
@media (min-width:1700px){html{font-size:18px}}
@media (max-width:520px){#hero h1{font-size:1.6rem}.cmp-name{max-width:45%}}
"""

# One script, delegated events. Inline in <head>.
HEAD = """
<script>
(function(){
  if (window.HM) return;
  var HM = window.HM = {};
  function closest(t, sel){ return t && t.closest ? t.closest(sel) : null; }
  document.addEventListener('pointerdown', function(ev){
    var st = closest(ev.target, '.stage'); if (st && closest(st, '.cmp').dataset.mode === 'hold'){ st.classList.add('holding'); try{ st.setPointerCapture(ev.pointerId) }catch(x){} }
    var fs = closest(ev.target, '.fs-stage'); if (fs && fs.dataset.mode === 'hold'){ fs.classList.add('holding') }
    var hb = closest(ev.target, '.fs-hold'); if (hb){ HM.fsStage().classList.add('holding') }
  });
  function release(ev){
    document.querySelectorAll('.stage.holding,.fs-stage.holding').forEach(function(n){ n.classList.remove('holding') });
  }
  document.addEventListener('pointerup', release); document.addEventListener('pointercancel', release);
  document.addEventListener('lostpointercapture', release);
  document.addEventListener('keydown', function(ev){
    var m = document.getElementById('hm-full');
    if (m && !m.hidden){
      if (ev.key === 'Escape'){ HM.fsClose(); return }
      if (ev.key === ' ' || ev.key === 'Enter'){ ev.preventDefault(); HM.fsStage().classList.add('holding') }
      return;
    }
    var st = closest(ev.target, '.stage');
    if (st && (ev.key === ' ' || ev.key === 'Enter') && closest(st, '.cmp').dataset.mode === 'hold'){ ev.preventDefault(); st.classList.add('holding') }
  });
  document.addEventListener('keyup', function(ev){ if (ev.key === ' ' || ev.key === 'Enter') release(ev) });
  document.addEventListener('input', function(ev){
    var r = closest(ev.target, '.range'); if (r){ closest(r, '.cmp').style.setProperty('--pos', r.value + '%') }
    var f = closest(ev.target, '.fs-range'); if (f){ HM.fsStage().style.setProperty('--fpos', f.value + '%') }
  });
  document.addEventListener('click', function(ev){
    var b = closest(ev.target, '.cmp-mode');
    if (b){ var c = closest(b, '.cmp'); var s = c.dataset.mode === 'slide'; c.dataset.mode = s ? 'hold' : 'slide';
      c.querySelector('.stage').classList.remove('holding');
      b.textContent = s ? 'Switch to slider' : 'Switch to hold';
      c.querySelector('.hint').textContent = s ? 'Press and hold the image to see the original' : 'Drag the slider: original on the left'; return }
    var cp = closest(ev.target, '.copy');
    if (cp){ var t = cp.dataset.text; function done(){ var o = cp.textContent; cp.textContent = 'Copied'; setTimeout(function(){ cp.textContent = o }, 1200) }
      function fb(){ var a = document.createElement('textarea'); a.value = t; a.style.position = 'fixed'; a.style.opacity = '0'; document.body.appendChild(a); a.select();
        try{ document.execCommand('copy'); done() }catch(x){} document.body.removeChild(a) }
      if (navigator.clipboard && window.isSecureContext){ navigator.clipboard.writeText(t).then(done, fb) } else { fb() } return }
    var fb2 = closest(ev.target, '.cmp-full'); if (fb2){ HM.fsOpen(fb2.dataset.name, fb2.dataset.prot, fb2.dataset.orig); return }
    if (closest(ev.target, '.fs-close')){ HM.fsClose(); return }
    var fm = closest(ev.target, '.fs-mode'); if (fm){ var s2 = HM.fsStage(); var sl = s2.dataset.mode === 'slide'; s2.dataset.mode = sl ? 'hold' : 'slide'; fm.textContent = sl ? 'Slider' : 'Hold'; fm.classList.toggle('on', !sl); return }
    var ft = closest(ev.target, '.fs-fit'); if (ft){ var s3 = HM.fsStage(); var full = s3.dataset.fit === 'full'; s3.dataset.fit = full ? 'fit' : 'full'; ft.textContent = full ? '100%' : 'Fit'; ft.classList.toggle('on', !full); return }
  });
  HM.fsStage = function(){ return document.querySelector('#hm-full .fs-stage') };
  HM.fsClose = function(){ var m = document.getElementById('hm-full'); if (m){ m.hidden = true; m.querySelectorAll('img').forEach(function(i){ i.removeAttribute('src') }) } document.body.style.overflow = '' };
  HM.fsOpen = function(name, prot, orig){
    var m = document.getElementById('hm-full');
    if (!m){ m = document.createElement('div'); m.id = 'hm-full'; document.body.appendChild(m);
      m.innerHTML = '<div class="fs-bar"><span class="fs-title"></span><button class="fs-hold" title="Hold to see the original (or hold Space)">Hold: original</button>'
        + '<button class="fs-mode">Slider</button><button class="fs-fit">100%</button><button class="fs-close">Close</button></div>'
        + '<div class="fs-note">Full resolution. Hold the button, the image or Space to see the original. Use 100% to inspect pixels, scroll to pan.</div>'
        + '<div class="fs-body"><div class="fs-stage" data-mode="hold" data-fit="fit"><img class="fs-m" alt=""><img class="fs-o" alt="">'
        + '<input class="fs-range" type="range" min="0" max="100" value="50" aria-label="Compare slider"><div class="fs-div"></div></div></div>' }
    var st = m.querySelector('.fs-stage'); st.dataset.mode = 'hold'; st.dataset.fit = 'fit'; st.classList.remove('holding');
    m.querySelector('.fs-mode').textContent = 'Slider'; m.querySelector('.fs-mode').classList.remove('on');
    m.querySelector('.fs-fit').textContent = '100%'; m.querySelector('.fs-fit').classList.remove('on');
    var mi = m.querySelector('.fs-m'), oi = m.querySelector('.fs-o'), title = m.querySelector('.fs-title');
    title.textContent = name + ' (loading)';
    mi.onload = function(){ title.textContent = name + '  ' + mi.naturalWidth + ' x ' + mi.naturalHeight + ' px' };
    mi.src = prot; oi.src = orig; m.hidden = false; document.body.style.overflow = 'hidden';
  };
})();
</script>
"""


def progress_bar(frac: float, message: str, sub: str = "") -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    return (f"<div class='prog'><div class='msg'>{e(message)}</div><div class='bar'><i style='width:{pct:.1f}%'></i></div>"
            f"<div class='sub'>{e(sub)}</div></div>")


def pill(kind: str, text: str) -> str:
    return f"<span class='pill {kind}'>{e(text)}</span>"


def card(it: dict) -> str:
    """One before/after card. `it` keys: name, full, orig_full, mist, orig, w, h, download, tag, target, m, seconds."""
    m = it.get("m")
    chips = []
    if it.get("tag") is not None:
        alt = it.get("alt") or it["tag"]
        chips.append(f"<span class='chip'>Tag: <b>{e(it['tag'])}</b><button type='button' class='copy' data-text=\"{e(alt)}\">Copy ALT</button></span>")
    if it.get("target"):
        chips.append(f"<span class='chip'>Pushes towards: <b>{e(it['target'])}</b></span>")
    if m:
        psnr = "inf" if m["psnr"] == float("inf") else f"{m['psnr']:.1f}"
        chips.append(f"<span class='stat'>largest change {m['max']:.0f}/255</span><span class='stat'>PSNR {psnr} dB</span>")
    if it.get("seconds") is not None:
        chips.append(f"<span class='stat'>{it['seconds']:.0f} s</span>")
    if it.get("fw"):
        chips.append(f"<span class='stat'>{it['fw']} x {it['fh']} px</span>")
    return f"""
<div class="cmp" data-mode="hold" style="--ar:{it['w']} / {it['h']}">
  <div class="cmp-head"><span class="cmp-name">{e(it['name'])}</span>
    <span class="cmp-btns"><button type="button" class="cmp-full" data-name="{e(it['name'])}" data-prot="file={e(it['full'])}" data-orig="file={e(it['orig_full'])}">Full size</button>
    <a class="cmp-dl" href="file={e(it['full'])}" download="{e(it['download'])}">Download</a></span></div>
  <div class="stage" tabindex="0" role="img" aria-label="Protected image. Press and hold, or hold Space, to see the original." oncontextmenu="return false">
    <img class="img-m" src="file={e(it['mist'])}" alt="" draggable="false">
    <img class="img-o" src="file={e(it['orig'])}" alt="" draggable="false">
    <span class="tag tag-m">Protected</span><span class="tag tag-o">Original</span>
    <input class="range" type="range" min="0" max="100" value="50" aria-label="Compare slider">
    <div class="divider"></div>
  </div>
  <div class="cmp-foot"><span class="hint">Press and hold the image to see the original</span>
    <button type="button" class="cmp-mode">Switch to slider</button></div>
  <div class="chips">{''.join(chips)}</div>
</div>"""


def cards(items) -> str:
    return '<div class="cmp-list">' + "".join(card(i) for i in items) + "</div>"
