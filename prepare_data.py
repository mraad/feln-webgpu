"""FELN.json + Layers.json -> data/{train,val}.jsonl (prompt/completion) + data/system_prompt.txt.

Schema lives in the weights (prior 7B runs: names-only prompt trained fastest and the
misses were column-identity errors that a column list, not hints, fixed). The prompt is
contract + layer names + column names: short enough for browser prefill, enough for the
model to spell columns right.
"""

import json
import random
from pathlib import Path

SRC = Path.home() / "Documents/ArcGIS/Projects/NorthSea"
OUT = Path(__file__).parent / "data"

CONTRACT = """Translate the question about the North Sea geodatabase into a JSON query plan.
Reply with ONLY a JSON object with keys "layers", "where", "relations":
"layers" lists layer names, first is the returned layer; "where" has one SQL WHERE clause per layer ("" if none);
"relations" has one spatial relation per secondary layer: within, contains, intersects, withinDistance <n> <unit>, notWithinDistance <n> <unit>."""


def system_prompt(layers: list[dict]) -> str:
    lines = [CONTRACT, ""]
    for layer in layers:
        cols = ", ".join(c["name"] for c in layer["columns"])
        lines.append(f"{layer['name']} ({layer['stype']}): {cols}")
    return "\n".join(lines)


def main() -> None:
    layers = json.loads((SRC / "Layers.json").read_text())["layers"]
    samples = json.loads((SRC / "FELN.json").read_text())
    system = system_prompt(layers)
    (OUT / "system_prompt.txt").write_text(system)

    rows = []
    for s in samples:
        m = s["meta"]
        assert len(m["where"]) == len(m["layers"]) and len(m["relations"]) == len(m["layers"]) - 1, s
        rows.append(
            {
                "prompt": [{"role": "system", "content": system}, {"role": "user", "content": s["text"]}],
                "completion": [{"role": "assistant", "content": json.dumps(m, separators=(",", ":"))}],
            }
        )

    # 10% holdout, stratified by relation arity so val covers 1/2/3-layer shapes.
    rng = random.Random(0)
    by_arity: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_arity.setdefault(len(s["meta"]["layers"]), []).append(i)
    val = set()
    for idx in by_arity.values():
        val.update(rng.sample(idx, len(idx) // 10))

    for name, keep in (("train", lambda i: i not in val), ("val", lambda i: i in val)):
        with (OUT / f"{name}.jsonl").open("w") as fh:
            n = 0
            for i, row in enumerate(rows):
                if keep(i):
                    fh.write(json.dumps(row) + "\n")
                    n += 1
        print(f"{name}: {n}")
    print(f"system prompt: {len(system)} chars")


if __name__ == "__main__":
    main()
