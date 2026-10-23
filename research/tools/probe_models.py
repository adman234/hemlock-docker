import sys, os, json, glob
sys.path.insert(0, r"C:/ns/ref")
import torch
home = os.path.join(os.path.expanduser("~"), ".nightshade")
out = {}
clip = torch.load(os.path.join(home, "clip_model.p"), map_location="cpu", weights_only=False)
v = clip.visual
out["clip"] = {"class": type(v).__name__, "input_resolution": v.input_resolution, "width": v.conv1.weight.shape[0], "patch": v.conv1.weight.shape[-1],
               "layers": len(v.transformer.resblocks), "embed_dim": clip.text_projection.shape[1], "context_length": clip.context_length, "vocab": clip.vocab_size}
blip = torch.load(os.path.join(home, "blip_model.pt"), map_location="cpu", weights_only=False)
c = blip.config
out["blip"] = {"vision_hidden": c.vision_config.hidden_size, "vision_layers": c.vision_config.num_hidden_layers, "image_size": c.vision_config.image_size,
               "text_hidden": c.text_config.hidden_size, "text_layers": c.text_config.num_hidden_layers, "name": getattr(c, "_name_or_path", None)}
import pickle
lp = pickle.load(open(os.path.join(home, "lpips_fn.p"), "rb"))
out["lpips"] = {"pnet_type": lp.pnet_type, "pnet_rand": lp.pnet_rand, "version": lp.version, "spatial": lp.spatial, "lpips": lp.lpips, "dtype": str(next(lp.parameters()).dtype)}
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import utils
    an = utils.ImageAnalyzer("cpu", home)
    out["mapping_entries"] = len(an.precompute_mapping)
out["pool"] = {"n": len(an.target_concepts), "first": an.target_concepts[:10]}
print("JSON " + json.dumps(out))

