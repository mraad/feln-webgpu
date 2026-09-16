"""FELN.json -> web/test/gold_sql.json: every plan with the SQL the Python compiler emits, so
`npm test` can check that feln_sql.js compiles identically.

    uv run gold_sql.py
"""

import json
from pathlib import Path

from feln import FELN, Layers, feln_to_sql

SRC = Path.home() / "Documents/ArcGIS/Projects/NorthSea"
OUT = Path(__file__).parent / "web/test/gold_sql.json"

catalog = Layers.load(SRC / "Layers.json")
rows = [{"feln": r["meta"], "sql": feln_to_sql(FELN(**r["meta"]), catalog)} for r in json.loads((SRC / "FELN.json").read_text())]
OUT.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
print(f"{len(rows)} plans -> {OUT}")
