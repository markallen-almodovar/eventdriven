"""
patch_pets_h5.py — Remove 'groups' key from DepthwiseConv2D configs in the pets h5 model.
This fixes compatibility between Keras 2.15 (which saved the model) and Keras 3 (current).
"""
import h5py
import json
import re
import shutil

src = "saved_model/pets_classification_model.h5"
dst = "saved_model/pets_classification_model_fixed.h5"

shutil.copy2(src, dst)
print(f"Copied {src} → {dst}")

with h5py.File(dst, "r+") as f:
    cfg_raw = f.attrs.get("model_config", None)
    if cfg_raw is None:
        print("ERROR: no model_config attribute found")
        exit(1)

    if isinstance(cfg_raw, bytes):
        cfg_str = cfg_raw.decode("utf-8")
    else:
        cfg_str = str(cfg_raw)

    original_len = len(cfg_str)

    # Remove "groups": 1 from DepthwiseConv2D configs
    cfg_str = re.sub(r',\s*"groups":\s*1', '', cfg_str)
    cfg_str = re.sub(r'"groups":\s*1,\s*', '', cfg_str)

    removed = original_len - len(cfg_str)
    print(f"Removed {removed} chars of 'groups' keys")

    f.attrs["model_config"] = cfg_str.encode("utf-8")

print("Patched successfully!")
print(f"Fixed model saved to: {dst}")
