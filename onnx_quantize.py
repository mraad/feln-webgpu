"""fp16 ONNX -> MatMulNBits weight-only quantization, fp16 activations.

    uv run onnx_quantize.py web/model/onnx/model_fp16.onnx web/model/onnx/model_q8_fp16.onnx [bits] [block]

Default 8-bit / block 128 / symmetric: in the browser it matched the fp16 graph on the holdout
(74/81 exact) at 443 MB vs 692 MB. 4-bit (block 32 or 128, symmetric or not) lost 3-6 plans,
mostly the third layer of 3-layer plans.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    bits, block = (int(a) for a in (sys.argv[3:5] or ("8", "128")))
    q = MatMulNBitsQuantizer(str(src), bits=bits, block_size=block, is_symmetric=True, accuracy_level=4)
    q.process()
    dst.with_name(dst.name + "_data").unlink(missing_ok=True)  # onnx.save appends to an existing data file
    onnx.save(q.model.model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=dst.name + "_data")
    print("saved", dst, f"{(dst.with_name(dst.name + '_data')).stat().st_size / 1e6:.0f} MB")
    # app.js keys its browser cache on this stamp, so a re-export never serves stale weights.
    version = dst.parent.parent / "version.txt"
    version.write_text(datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    print("version", version.read_text())


if __name__ == "__main__":
    main()
