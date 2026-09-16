"""Put fine-tuned LFM2-350M weights into onnx-community's LFM2-350M ONNX graph.

optimum-onnx has no lfm2 exporter, but the fine-tuned model has the exact architecture of
LiquidAI/LFM2-350M, so the exported graph is reusable: only the initializers change.
Mapping initializer -> HF parameter is derived from the BASE weights by value (shape +
allclose, transposed or not), so it needs no knowledge of the exporter's naming.

    uv run onnx_swap.py --base-onnx scratch/model_fp16.onnx --model out/lfm2-350m-feln --out web/model/onnx/model_fp16.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors.torch import load_file


def load_params(model_dir: str) -> dict[str, torch.Tensor]:
    params = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        params.update(load_file(str(f)))
    if not any(k.startswith("lm_head") for k in params):  # tied embeddings
        params["lm_head.weight"] = params["model.embed_tokens.weight"]
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-onnx", required=True)
    ap.add_argument("--base", default="LiquidAI/LFM2-350M")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    base = load_params(snapshot_download(args.base, allow_patterns=["*.safetensors"]))
    tuned = load_params(args.model)
    assert base.keys() == tuned.keys(), set(base) ^ set(tuned)
    base_np = {k: v.float().numpy() for k, v in base.items()}
    by_shape: dict[tuple, list[str]] = {}
    for k, v in base_np.items():
        by_shape.setdefault(v.shape, []).append(k)

    m = onnx.load(args.base_onnx)
    mapped, unmapped, unchanged = 0, [], 0
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size < 1024:  # scalars, shapes, rope tables
            continue
        hit = None
        for transposed, shape in ((False, arr.shape), (True, arr.shape[::-1])):
            for name in by_shape.get(shape, []):
                cand = base_np[name].T if transposed else base_np[name]
                if np.allclose(cand.astype(np.float16), arr.astype(np.float16), atol=1e-3, rtol=1e-2):
                    hit = (name, transposed)
                    break
            if hit:
                break
        if hit is None:
            unmapped.append((init.name, arr.shape, arr.dtype))
            continue
        name, transposed = hit
        new = tuned[name].float().numpy()
        new = new.T if transposed else new
        if np.array_equal(new.astype(arr.dtype), arr):
            unchanged += 1
        init.CopyFrom(numpy_helper.from_array(new.astype(arr.dtype), init.name))
        mapped += 1
    print(f"mapped {mapped} initializers ({unchanged} unchanged), unmapped {len(unmapped)}")
    for u in unmapped:
        print("  UNMAPPED", u)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_name(out.name + "_data").unlink(missing_ok=True)  # onnx.save appends to an existing data file
    onnx.save(m, str(out), save_as_external_data=True, all_tensors_to_one_file=True, location=out.name + "_data")
    print("saved", out)


if __name__ == "__main__":
    main()
