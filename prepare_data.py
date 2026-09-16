"""FELN.json + Layers.json -> data/{train,val}.jsonl (prompt/completion) + data/system_prompt.txt.

Schema lives in the weights (prior 7B runs: names-only prompt trained fastest and the
misses were column-identity errors that a column list, not hints, fixed). The prompt is
contract + layer names + column names: short enough for browser prefill, enough for the
model to spell columns right.
"""

import argparse
import json
import random
import re
from pathlib import Path

SRC = Path.home() / "Documents/ArcGIS/Projects/NorthSea"
OUT = Path(__file__).parent / "data"

CONTRACT = """Translate the question about the North Sea geodatabase into a JSON query plan.
Reply with ONLY a JSON object with keys "layers", "where", "relations":
"layers" lists layer names, first is the returned layer; "where" has one SQL WHERE clause per layer ("" if none);
"relations" has one spatial relation per secondary layer: within, contains, intersects, withinDistance <n> <unit>, notWithinDistance <n> <unit>."""


def system_prompt(layers: list[dict], okf: Path | None) -> str:
    """Contract + schema. With --okf the schema is the OKF bundle verbatim (index + one doc per
    layer, ~10k tokens); without it, one line of column names per layer (~300 tokens)."""
    lines = [CONTRACT, ""]
    if okf:
        lines.append((okf / "index.md").read_text().strip())
        for layer in layers:
            doc = okf / f"{layer['name']}.md"
            lines += ["", doc.read_text().strip()]
        # the OKF docs embed absolute file:// paths under the author's home; ship them as /Users/user (a ~ costs one holdout question)
        return "\n".join(lines).replace(str(Path.home()), "/Users/user")
    for layer in layers:
        cols = ", ".join(c["name"] for c in layer["columns"])
        lines.append(f"{layer['name']} ({layer['stype']}): {cols}")
    return "\n".join(lines)


# FELN.json mentions water_depth in 5 of 1000 rows, always as "water depth", so the model copied a
# user's bare "depth" into a non-existent column. These templates teach the synonyms; thresholds
# are drawn from the real value range (NorthSea water_depth is 0-1350 m).
DEPTH_TEMPLATES = [
    ("Show all wells with depth > {x} meters", ">"),
    ("Show all wells with depth greater than {x} m", ">"),
    ("Find wells deeper than {x} meters", ">"),
    ("List wells with a depth of more than {x} meters", ">"),
    ("Wells with water depth over {x}", ">"),
    ("Which wells are in water deeper than {x} meters?", ">"),
    ("Show wells with depth < {x} meters", "<"),
    ("Find wells shallower than {x} meters", "<"),
    ("List wells with depth under {x} m", "<"),
    ("Wells in water less than {x} meters deep", "<"),
    ("Show all wells with depth >= {x} meters", ">="),
    ("Find wells at least {x} meters deep", ">="),
    ("Show wells with depth <= {x} meters", "<="),
    ("Find wells at most {x} meters deep", "<="),
    ("Show wells whose depth is {x} meters", "="),
]


def depth_rows(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        template, op = DEPTH_TEMPLATES[i % len(DEPTH_TEMPLATES)]
        x = rng.choice([rng.randint(50, 1300), round(rng.uniform(50, 1300), 1)])
        out.append({"text": template.format(x=x), "meta": {"layers": ["Wells"], "where": [f'"water_depth" {op} {x}'], "relations": []}})
    return out


# The grammar states every spatial constraint as its own sentence ("The returned wells must be
# within 4 miles of ...") and names pipeline phase as "where current phase is 'IN SERVICE'". Users
# write "and are within 4 miles of an in-service pipeline"; the model then folded the constraint
# into the first layer's WHERE. These rewrites keep the plan and change only the wording.
CONSTRAINT_RE = re.compile(r"\. The returned \w+ must (be |)")
PHASES = {"IN SERVICE": "in-service", "DECOMMISSIONED": "decommissioned", "ABANDONED IN PLACE": "abandoned"}
PHASE_RE = re.compile(r"pipelines where current phase is '(IN SERVICE|DECOMMISSIONED|ABANDONED IN PLACE)'")


def conjoin(text: str, rng: random.Random) -> str | None:
    """'Show A. The returned wells must be within X of B.' -> 'Show A that are within X of B.'

    The first constraint joins with "that", later ones with "and": "wells that are within 4 miles of
    pipelines and contain one or more ...". "must be" -> "are"; other verbs keep their form.
    """
    if not CONSTRAINT_RE.search(text):
        return None
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        word = "that" if n == 1 else "and"
        return f" {word} are " if m.group(1) else f" {word} "

    return CONSTRAINT_RE.sub(repl, text)


def phase_adjective(text: str) -> str | None:
    """"pipelines where current phase is 'IN SERVICE'" -> 'in-service pipelines'."""
    if not PHASE_RE.search(text):
        return None
    return PHASE_RE.sub(lambda m: f"{PHASES[m.group(1)]} pipelines", text)


PHASE_TEMPLATES = ["Show {adj} pipelines", "List all {adj} pipelines", "Find the {adj} pipelines", "Which pipelines are {adj}?"]


def phase_rows() -> list[dict]:
    return [
        {"text": t.format(adj=adj), "meta": {"layers": ["Pipelines"], "where": [f'"current_phase" = \'{code}\''], "relations": []}}
        for code, adj in PHASES.items()
        for t in PHASE_TEMPLATES
    ]


def paraphrases(samples: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for s in samples:
        variants = [s["text"]]
        for fn in (lambda t: phase_adjective(t), lambda t: conjoin(t, rng)):
            for v in list(variants):
                new = fn(v)
                if new and new not in variants:
                    variants.append(new)
        # Every phase rewrite is kept (28 rows); conjoined constraints for 40% of multi-layer rows.
        for v in variants[1:]:
            if "current phase" not in s["text"] and rng.random() > 0.4:
                continue
            out.append({"text": v, "meta": s["meta"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--okf", type=Path, default=None, help="OKF bundle dir to embed in the system prompt (e.g. NorthSea/okf)")
    args = ap.parse_args()
    layers = json.loads((SRC / "Layers.json").read_text())["layers"]
    samples = json.loads((SRC / "FELN.json").read_text())
    n_source = len(samples)
    # FELN.json rows carry the humanized text plus the grammar text it came from (regen_northsea.py);
    # train on both wordings, and derive the rule-based paraphrases from the grammar text.
    # 10% holdout of the source plans, stratified by number of layers. A plan's grammar wording and
    # the rule-based paraphrases derived from it stay on the same side as its humanized text.
    rng = random.Random(0)
    by_arity: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_arity.setdefault(len(s["meta"]["layers"]), []).append(i)
    val_source: set[int] = set()
    for idx in by_arity.values():
        val_source.update(rng.sample(idx, len(idx) // 10))
    grammar = {i: {"text": s["source_text"], "meta": s["meta"]} for i, s in enumerate(samples) if s.get("source_text") and s["source_text"] != s["text"]}
    val_grammar = [g for i, g in grammar.items() if i in val_source]
    train_grammar = [g for i, g in grammar.items() if i not in val_source]
    extra = depth_rows(random.Random(1), 45) + phase_rows()
    samples += val_grammar + train_grammar + extra + paraphrases(train_grammar or [s for i, s in enumerate(samples) if i not in val_source], random.Random(2))
    val = val_source | set(range(n_source, n_source + len(val_grammar))) | set(range(n_source + len(grammar), n_source + len(grammar) + len(extra), 5))
    system = system_prompt(layers, args.okf)
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


    for name, keep in (("train", lambda i: i not in val), ("val", lambda i: i in val)):
        with (OUT / f"{name}.jsonl").open("w") as fh:
            n = 0
            for i, row in enumerate(rows):
                if keep(i):
                    fh.write(json.dumps(row) + "\n")
                    n += 1
        print(f"{name}: {n}")
    print(f"system prompt: {len(system)} chars ({'OKF' if args.okf else 'column names'})")


if __name__ == "__main__":
    main()
