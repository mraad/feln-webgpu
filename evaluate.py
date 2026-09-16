"""Score FELN predictions on a JSONL split.

    uv run evaluate.py data/val.jsonl --model LiquidAI/LFM2-350M      # HF model, greedy decode
    uv run evaluate.py data/val.jsonl --preds preds.jsonl              # pre-generated {"text": ...} per row

Metrics: valid JSON, exact match (normalised), FELNCompare.structural, SQL compiles, and
OBJECTID Jaccard against gold when both execute on NorthSea.ddb (stats tables are not in
the DDB and are skipped for execution).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb
from feln import FELN, FELNCompare, Layers, feln_to_sql

SRC = Path.home() / "Documents/ArcGIS/Projects/NorthSea"


def parse_plan(text: str) -> dict | None:
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip())
    start = text.find("{")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            try:
                obj = json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                return None
            return obj if isinstance(obj, dict) else None
    return None


def norm(plan: dict) -> dict:
    return {
        "layers": [str(x) for x in plan.get("layers", [])],
        "where": [re.sub(r"\s+", " ", str(x)).strip() for x in plan.get("where", [])],
        "relations": [re.sub(r"\s+", " ", str(x)).strip() for x in plan.get("relations", [])],
    }


def to_feln(plan: dict) -> FELN | None:
    try:
        return FELN(**norm(plan))
    except Exception:
        return None


def generate_hf(model_id: str, rows: list[dict], max_new_tokens: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("mps").eval()
    outs = []
    for i, row in enumerate(rows):
        enc = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("mps")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        outs.append(tok.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True))
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", file=sys.stderr)
    return outs


def score(rows: list[dict], preds: list[str], con: duckdb.DuckDBPyConnection | None, catalog: Layers) -> dict:
    tally = {"valid": 0, "exact": 0, "structural": 0.0, "sql_ok": 0, "exec_n": 0, "jaccard": 0.0}
    misses = []
    for row, out in zip(rows, preds, strict=True):
        gold = json.loads(row["completion"][0]["content"])
        q = row["prompt"][-1]["content"]
        plan = parse_plan(out) if out else None
        pred = to_feln(plan) if plan else None
        if pred is None:
            misses.append({"q": q, "raw": out[:300], "gold": gold})
            continue
        tally["valid"] += 1
        gold_f = FELN(**norm(gold))
        tally["structural"] += FELNCompare.structural(gold_f, pred)
        if norm(plan) == norm(gold):
            tally["exact"] += 1
        else:
            misses.append({"q": q, "pred": norm(plan), "gold": norm(gold)})
        try:
            sql = feln_to_sql(pred, catalog)
            tally["sql_ok"] += 1
        except Exception:
            continue
        if con is not None and all(catalog.find_layer(n) and con.sql(f"select count(*) from information_schema.tables where table_name='{catalog.find_layer(n).table_name or n}'").fetchone()[0] for n in pred.layers + gold_f.layers):
            try:
                p = {r[0] for r in con.sql(sql).fetchall()}
                g = {r[0] for r in con.sql(feln_to_sql(gold_f, catalog)).fetchall()}
            except Exception as e:  # SQL parsed but did not run: counts as 0 overlap
                misses[-1:] = misses[-1:] if norm(plan) != norm(gold) else []
                misses.append({"q": q, "sql_error": str(e)[:200], "pred": norm(plan)})
                tally["exec_n"] += 1
                continue
            tally["exec_n"] += 1
            tally["jaccard"] += 1.0 if not (p | g) else len(p & g) / len(p | g)
    return tally, misses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--preds", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--save-preds", type=Path)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.split.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if args.preds:
        preds = [json.loads(l)["text"] for l in args.preds.read_text().splitlines()][: len(rows)]
    else:
        preds = generate_hf(args.model, rows, args.max_new_tokens)
        if args.save_preds:
            args.save_preds.write_text("".join(json.dumps({"text": p}) + "\n" for p in preds))

    catalog = Layers.load(SRC / "Layers.json")
    con = duckdb.connect(str(SRC / "NorthSea.ddb"), read_only=True)
    con.sql("load spatial")
    tally, misses = score(rows, preds, con, catalog)
    n = len(rows)
    print(f"\n=== {args.split.name} n={n} {args.model or args.preds} ===")
    print(f"  valid      {tally['valid']}/{n}")
    print(f"  exact      {tally['exact']}/{n}  {tally['exact'] / n:.1%}")
    print(f"  structural {tally['structural'] / n:.3f}")
    print(f"  sql_ok     {tally['sql_ok']}/{n}")
    print(f"  exec       {tally['exec_n']}  jaccard {tally['jaccard'] / max(tally['exec_n'], 1):.3f}")
    for m in misses[:25]:
        print("--- MISS ---")
        print(json.dumps(m, ensure_ascii=False)[:600])
    print(f"({len(misses)} misses total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
