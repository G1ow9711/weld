"""Fetch a pinned, licensed SpUNet source and record the Windows adapters."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "ad653ee75857f57a44cc0ee7cea5fd0372908b4d"
BASE = f"https://raw.githubusercontent.com/Pointcept/Pointcept/{COMMIT}/"
REL = "pointcept/models/sparse_unet/spconv_unet_v1m1_base.py"


def fetch(relative):
    request = urllib.request.Request(BASE + relative, headers={"User-Agent": "weld-training-setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    dest = ROOT / "training/third_party/pointcept"
    dest.mkdir(parents=True, exist_ok=True)
    original = fetch(REL)
    license_text = fetch("LICENSE")
    replacements = {
        "from torch_geometric.utils import scatter": "from training.models.compat import scatter",
        "from timm.models.layers import trunc_normal_": "from training.models.compat import trunc_normal_",
        "from pointcept.models.builder import MODELS": "from training.models.compat import MODELS",
        "from pointcept.models.utils import offset2batch": "from training.models.compat import offset2batch",
    }
    adapted = original.decode("utf-8")
    for before, after in replacements.items():
        if adapted.count(before) != 1:
            raise RuntimeError(f"Unexpected upstream source: {before}")
        adapted = adapted.replace(before, after)
    adapted_bytes = adapted.encode("utf-8")
    (dest / "original_spconv_unet_v1m1_base.py").write_bytes(original)
    (dest / "LICENSE").write_bytes(license_text)
    model = ROOT / "training/models/spunet.py"
    model.write_bytes(adapted_bytes)
    record = {
        "repository": "https://github.com/Pointcept/Pointcept",
        "tag": "v1.5.2", "commit": COMMIT, "source_path": REL,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "adapted_sha256": hashlib.sha256(adapted_bytes).hexdigest(),
        "changes": replacements,
        "scope": "Standalone SpUNet backbone; not the complete Pointcept engine.",
    }
    (dest / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
