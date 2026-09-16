"""Regenerate NorthSea's Layers.json, OKF bundle and FELN.json, humanize the FELN texts, validate.

    uv run regen_northsea.py [--n 1000] [--seed 20260916] [--batch 25] [--no-humanize]

Steps (each can be re-run; sources are never edited):
  1. back up Layers.json, FELN.json, feln-generated.jsonl, okf/ -> backups/regen-<utc>/
  2. Layers.json <- `layers-json NorthSea.aprx --use-ilike` (GDAL reader), then: keep only the
     columns the previous catalog exposed (Pro hid symbol, casing_lot, ... in the map), reapply
     feln-liquid's presentation refinements (aliases, boolean wording, water-depth unit)
  3. okf/ <- `layers-okf NorthSea.aprx --use-ilike`
  4. FELN.json <- feln.generate, deduplicated on the normalized plan, audited with feln-liquid's audit()
  5. texts humanized in batches through `claude -p`; every literal, number and unit of the grammar
     text must survive, texts must stay unique; failures keep the grammar text (counted)
  6. FELN.json / feln-generated.jsonl / regen-validation-<date>.json written
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from feln import Layers
from feln.generate import generate, records_to_jsonl
from feln.identical import normalize_where

SRC = Path.home() / "Documents/ArcGIS/Projects/NorthSea"
LAYERS_JSON_BIN = Path.home() / "GWorkspace/layers-json/.venv/bin"
sys.path.insert(0, str(Path.home() / "GWorkspace/feln-liquid"))
from sync_northsea import audit, refine_catalog  # noqa: E402

HUMANIZE_PROMPT = """You rewrite synthetic database requests so they read like a real person typed them.

Rules, all mandatory:
- Keep the exact meaning: same conditions, same AND/OR/NOT logic, same layers, same spatial relation.
- Keep every literal exactly as written: quoted values, names, codes, numbers, distances and their units.
  Do not round numbers, convert units, or drop quotes around values that were quoted.
- Vary the style across the list: questions, commands, short casual phrasing, longer sentences,
  "which ...", "I need ...", "give me ...". Do not start every item the same way.
- Keep the domain words that identify the layer (wells, pipelines, discoveries) and the
  classifications (e.g. dry wells, gas pipelines, oil discoveries, in-service pipelines).
- One rewrite per input, same order, no numbering, no explanations.

Reply with ONLY a JSON array of strings, one per input, nothing else.

Inputs:
"""

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
QUOTED_RE = re.compile(r"'([^']+)'")
UNIT_RE = re.compile(r"\b(meters?|kilometers?|miles?|feet|foot)\b")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def backup() -> Path:
    dest = SRC / "backups" / datetime.now(UTC).strftime("regen-%Y%m%d-%H%M%S-UTC")
    dest.mkdir(parents=True)
    for name in ("Layers.json", "FELN.json", "feln-generated.jsonl"):
        if (SRC / name).exists():
            shutil.copy2(SRC / name, dest / name)
    if (SRC / "okf").is_dir():
        shutil.copytree(SRC / "okf", dest / "okf")
    return dest


def regen_layers(previous: dict) -> tuple[dict, list[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        run([str(LAYERS_JSON_BIN / "layers-json"), str(SRC / "NorthSea.aprx"), "-o", tmp, "--use-ilike"])
        raw = json.loads((Path(tmp) / "Layers.json").read_text())
    notes = []
    prev = {layer["name"]: layer for layer in previous["layers"]}
    for layer in raw["layers"]:
        old = prev.get(layer["name"])
        if not old:
            notes.append(f"new layer {layer['name']}")
            continue
        visible = {c["name"] for c in old["columns"]}
        hidden = [c["name"] for c in layer["columns"] if c["name"] not in visible]
        layer["columns"] = [c for c in layer["columns"] if c["name"] in visible]
        if hidden:
            notes.append(f"{layer['name']}: dropped columns hidden in the previous catalog: {', '.join(hidden)}")
    dropped = [n for n in prev if n not in {layer["name"] for layer in raw["layers"]}]
    if dropped:
        notes.append(f"layers not produced by layers-json (Pro-toolbox tables): {', '.join(dropped)}")
    notes += refine_catalog(raw)
    return raw, notes


def regen_okf() -> list[str]:
    out = SRC / "okf"
    if out.exists():
        shutil.rmtree(out)
    run([str(LAYERS_JSON_BIN / "layers-okf"), str(SRC / "NorthSea.aprx"), "-o", str(out), "--use-ilike"])
    return sorted(p.name for p in out.glob("*.md"))


def regen_feln(catalog: Layers, n: int, seed: int) -> list[dict]:
    records, seen = [], set()
    for row in generate(catalog, int(n * 1.3), seed=seed, alias_suffix=True, layer_only=0.25, normalize=False):
        key = json.dumps({**row["meta"], "where": [normalize_where(w) for w in row["meta"]["where"]]}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            records.append({"text": row["text"], "meta": row["meta"]})
        if len(records) == n:
            break
    assert len(records) == n, len(records)
    return records


def claude_rewrite(texts: list[str]) -> list[str]:
    prompt = HUMANIZE_PROMPT + json.dumps(texts, ensure_ascii=False, indent=0)
    res = run(["claude", "-p", prompt, "--setting-sources", "", "--tools", "", "--model", "claude-opus-5", "--output-format", "json"])
    reply = json.loads(res.stdout)["result"].strip()
    reply = reply[reply.index("[") : reply.rindex("]") + 1]
    out = json.loads(reply)
    if not (isinstance(out, list) and len(out) == len(texts) and all(isinstance(t, str) for t in out)):
        raise ValueError(f"bad rewrite shape: {reply[:200]}")
    return [t.strip() for t in out]


def faithful(source: str, text: str) -> list[str]:
    """Why a rewrite is unacceptable; empty when every literal, number and unit survived."""
    low = text.lower()
    problems = []
    for q in QUOTED_RE.findall(source):
        if q.lower() not in low:
            problems.append(f"missing literal {q!r}")
    for num in NUM_RE.findall(source):
        if num not in text:
            problems.append(f"missing number {num}")
    for unit in set(UNIT_RE.findall(source)):
        stem = unit.rstrip("s").replace("feet", "f").replace("foot", "f")
        if stem not in low:
            problems.append(f"missing unit {unit}")
    if not text or text.lower() == source.lower():
        problems.append("unchanged")
    return problems


def humanize(records: list[dict], batch: int) -> dict:
    stats = {"rewritten": 0, "kept_grammar": 0, "retried": 0, "problems": []}
    seen = set()
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        sources = [r["text"] for r in chunk]
        try:
            rewrites = claude_rewrite(sources)
        except Exception as err:  # one bad batch must not lose the run
            stats["problems"].append(f"batch {start}: {err}")
            rewrites = list(sources)
        for rec, new in zip(chunk, rewrites, strict=True):
            problems = faithful(rec["text"], new)
            if problems or new in seen:
                stats["retried"] += 1
                try:
                    new = claude_rewrite([rec["text"]])[0]
                    problems = faithful(rec["text"], new)
                except Exception as err:
                    problems = [str(err)]
            rec["source_text"] = rec["text"]
            if problems or new in seen:
                stats["kept_grammar"] += 1
                stats["problems"].append(f"{rec['text'][:80]} -> {new[:80]}: {problems or 'duplicate'}")
            else:
                rec["text"] = new
                stats["rewritten"] += 1
            seen.add(rec["text"])
        print(f"  humanized {min(start + batch, len(records))}/{len(records)}", file=sys.stderr, flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--no-humanize", action="store_true")
    args = ap.parse_args()

    dest = backup()
    print("backup:", dest)
    previous = json.loads((SRC / "Layers.json").read_text())
    raw, notes = regen_layers(previous)
    (SRC / "Layers.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
    print("Layers.json:", [layer["name"] for layer in raw["layers"]])
    for note in notes:
        print("  ", note)
    print("okf:", regen_okf())

    catalog = Layers.model_validate(raw)
    records = regen_feln(catalog, args.n, args.seed)
    report = audit(records, catalog)
    print("audit:", report["checks"])
    stats = humanize(records, args.batch) if not args.no_humanize else {}
    assert len({r["text"] for r in records}) == len(records)
    (SRC / "FELN.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    (SRC / "feln-generated.jsonl").write_text(records_to_jsonl(records))
    report.update(seed=args.seed, catalog_notes=notes, humanize=stats, backup=str(dest),
                  generated_at=datetime.now(UTC).isoformat(timespec="seconds"))
    out = SRC / f"regen-validation-{datetime.now(UTC):%Y%m%d}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("humanize:", {k: v for k, v in stats.items() if k != "problems"}, f"({len(stats.get('problems', []))} problems)")
    print("validation:", out)


if __name__ == "__main__":
    main()
