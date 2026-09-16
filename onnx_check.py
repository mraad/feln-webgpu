"""Greedy-decode val prompts with the swapped ONNX graph (onnxruntime) and with the HF
model, and require identical text. Proves the initializer swap in onnx_swap.py.

    uv run onnx_check.py --onnx web/model/onnx/model_fp16.onnx --model out/lfm2-350m-feln --n 5
"""

import argparse
import json

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ort_generate(sess, tok, prompt_ids: list[int], max_new: int, dtype) -> str:
    names = [i.name for i in sess.get_inputs()]
    state = {}
    for i in sess.get_inputs():
        if i.name.startswith("past_conv"):
            state[i.name] = np.zeros((1, 1024, 3), dtype=dtype)
        elif i.name.startswith("past_key_values"):
            state[i.name] = np.zeros((1, 8, 0, 64), dtype=dtype)
    ids = list(prompt_ids)
    step_ids = ids
    out_ids = []
    for _ in range(max_new):
        feeds = {
            "input_ids": np.array([step_ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "num_logits_to_keep": np.array(1, dtype=np.int64),
            **state,
        }
        outs = sess.run(None, feeds)
        out_names = [o.name for o in sess.get_outputs()]
        res = dict(zip(out_names, outs))
        nxt = int(res["logits"][0, -1].argmax())
        for k in list(state):
            state[k] = res[k.replace("past_conv", "present_conv").replace("past_key_values", "present")]
        out_ids.append(nxt)
        ids.append(nxt)
        step_ids = [nxt]
        if nxt == tok.eos_token_id:
            break
    return tok.decode(out_ids, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    dtype = np.float16 if "float16" in sess.get_inputs()[3].type else np.float32
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    rows = [json.loads(l) for l in open("data/val.jsonl")][: args.n]
    same = 0
    for row in rows:
        enc = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            g = hf.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
        ref = tok.decode(g[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        got = ort_generate(sess, tok, enc["input_ids"][0].tolist(), args.max_new, dtype)
        same += got == ref
        print("OK " if got == ref else "DIFF", got[:160])
        if got != ref:
            print("  HF:", ref[:160])
    print(f"{same}/{len(rows)} identical")


if __name__ == "__main__":
    main()
