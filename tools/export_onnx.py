"""Export a HuggingFace encoder to ONNX with dynamic batch/seq axes.

Why ONNX and not a direct TorchScript -> OpenVINO conversion: the hs-embed-1 /
ModernBERT family carries a ~256k-row embedding table; ov.convert_model's
TorchScript trace chokes on it ("Cannot insert a Tensor that requires grad as a
constant" and constant-blob size limits). The legacy ONNX exporter handles it
cleanly. Keep dynamo=False — the new exporter needs onnxscript and gained us
nothing for this static graph.

Usage (inside the service image, which already has torch+transformers):
    python3 tools/export_onnx.py --src /models/hs-embed-1 --out /models-out

The output directory is directly servable by MODEL_FORMAT=onnx: model.onnx plus
the tokenizer files and 1_Pooling/config.json copied from the source directory.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]
POOLING_DIR = "1_Pooling"


def export(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(src), local_files_only=True)
    model = AutoModel.from_pretrained(str(src), local_files_only=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)  # freeze weights so export embeds them as constants

    enc = tok(["export probe"], padding=True, return_tensors="pt")
    target = out / "model.onnx"
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(target),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"exported: {target}")

    for name in TOKENIZER_FILES:
        if (src / name).is_file():
            shutil.copy(src / name, out / name)
    if (src / POOLING_DIR).is_dir():
        shutil.copytree(src / POOLING_DIR, out / POOLING_DIR, dirs_exist_ok=True)
    if (src / "config.json").is_file():
        shutil.copy(src / "config.json", out / "config.json")
    print(f"tokenizer + pooling config copied to {out}")

    # Sanity: OpenVINO must read the file and produce finite output.
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(target)), "CPU")
    result = compiled(
        {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
    )[0]
    assert np.isfinite(result).all(), "exported graph produced non-finite output"
    print(f"ov sanity: shape {result.shape} finite=True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="HF model directory (safetensors)")
    parser.add_argument("--out", type=Path, required=True, help="output directory for model.onnx")
    args = parser.parse_args()
    export(args.src, args.out)
