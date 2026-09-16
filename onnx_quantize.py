"""fp16 ONNX -> MatMulNBits weight-only quantization, fp16 activations.

    uv run onnx_quantize.py web/model/onnx/model_fp16.onnx web/model/onnx/model_q8_fp16.onnx [bits] [block]

Default 8-bit / block 128 / symmetric: in the browser it matched the fp16 graph on the holdout
(74/81 exact) at 443 MB vs 692 MB. 4-bit (block 32 or 128, symmetric or not) lost 3-6 plans,
mostly the third layer of 3-layer plans.
"""

import sys
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    bits, block = (int(a) for a in (sys.argv[3:5] or ("8", "128")))
    q = MatMulNBitsQuantizer(str(src), bits=bits, block_size=block, is_symmetric=True, accuracy_level=4)
    q.process()
    onnx.save(q.model.model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=dst.name + "_data")
    print("saved", dst, f"{(dst.with_name(dst.name + '_data')).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
