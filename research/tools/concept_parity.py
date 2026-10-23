"""Parity of concept and target SELECTION RULES: original ImageAnalyzer vs nsre.concept.

usage: python -I concept_parity.py REFDIR NSRE_SRC CORPUS_DIR   (CPU; USERPROFILE = scratch home with assets)
Uses the original's own BLIP/CLIP/tagger objects as inputs so only the selection logic is compared.
No proprietary asset content is printed beyond counts and match flags.
"""
import contextlib
import glob
import io
import json
import os
import sys

import nltk
import numpy as np
import torch

refdir, nsre_src, corpus = sys.argv[1:4]
sys.path[:0] = [refdir, nsre_src]
home = os.path.join(os.path.expanduser("~"), ".nightshade")


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


import utils  # noqa: E402
from nsre import concept as C  # noqa: E402
from nsre.imageio import load_img  # noqa: E402

an = quiet(utils.ImageAnalyzer, "cpu", home)
res = {"concept": [], "target_unmapped": [], "target_mapped": []}

# 1. concept detection on the corpus (same caption, same CLIP, our rules)
for f in sorted(glob.glob(os.path.join(corpus, "*.png")) + glob.glob(os.path.join(corpus, "*.jpg"))):
    img = load_img(f)
    if img is None:
        continue
    ref_c = quiet(an.get_current_concept, img)
    cap = quiet(an.get_blip_caption, img)
    nouns = C.extract_nouns(cap, nltk.pos_tag)
    ours = C.pick_concept(nouns, lambda prompts, _i=img: quiet(an.clip_model, _i, prompts, True))
    res["concept"].append({"file": os.path.basename(f), "caption": cap, "ref": ref_c, "ours": ours, "match": ref_c == ours})

# 2. target selection, words NOT in precompute_mapping -> similarity ranks 4..9 with seed 678121
words = ("table chair window tree river bridge guitar piano clock book pencil street beach desert island castle tower garden "
         "bicycle train airplane tiger lion bird fish butterfly mushroom coffee bread cheese pizza hat shoe glove ring "
         "mirror lamp curtain pillow sofa kitchen forest").split()
pool, embs = an.target_concepts, an.target_concepts_embs.astype(np.float32)
for w in words:
    if w in an.precompute_mapping:
        continue
    ref_t = quiet(an.get_target_style, w)
    e = quiet(an.clip_model.text_emb, w).cpu().numpy().astype(np.float32).reshape(1, -1)
    sim = (e / np.linalg.norm(e)) @ (embs / np.linalg.norm(embs, axis=1, keepdims=True)).T
    ours = C.pick_target(w, pool, sim)
    res["target_unmapped"].append({"word": w, "match": ref_t == ours})

# 3. mapped words: mapping wins
for w in list(an.precompute_mapping)[:60]:
    res["target_mapped"].append({"match": quiet(an.get_target_style, w) == C.pick_target(w, pool, np.zeros(len(pool)), mapping=an.precompute_mapping)})

summary = {k: {"n": len(v), "matches": sum(1 for x in v if x["match"])} for k, v in res.items()}
print("JSON " + json.dumps({"summary": summary, "concepts": res["concept"],
                            "unmapped_mismatches": [x["word"] for x in res["target_unmapped"] if not x["match"]]}))
