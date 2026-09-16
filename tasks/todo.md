# feln-webgpu: text -> FELN -> DuckDB, fully in browser

Phase 1 (done, commits ab53b25..810689d): LFM2-350M fine-tune, ONNX swap, 8-bit, duckdb-wasm, ArcGIS 5.1 map, v3 data.

## Phase 2: regenerate NorthSea artifacts, humanize, retrain with OKF (2026-09-16)

- [x] 1. `regen_northsea.py`: back up Layers.json / FELN.json / feln-generated.jsonl / okf to backups/regen-<ts>
- [x] 2. Layers.json from NorthSea.aprx via `layers-json` (GDAL), keep prior column visibility, reapply feln-liquid `refine_catalog` aliases
- [x] 3. OKF via `layers-okf` into okf/
- [x] 4. FELN.json: `feln.generate` 1000 (seed 20260916, alias_suffix, layer_only 0.25), dedupe, `audit`
- [x] 5. Humanize texts with `claude -p` in batches; keep `source_text`; validate literals/labels/units present, uniqueness, meta unchanged; write validation JSON
- [x] 6. prepare_data.py `--okf`: system prompt = contract + OKF; max_length raised
- [x] 7. Train on the RTX box, copy back, evaluate on holdout
- [x] 8. Export (swap, 8-bit), serve, browser test incl. long-prompt latency

- [x] 9. LFM2.5-1.2B-Instruct (feln-liquid model) trained on GPU 1 with the same data; evaluate, decide on browser export (~1.5 GB 8-bit)

## Decisions
- Base model LFM2-350M; map ArcGIS JS SDK 5.1; browser model 8-bit MatMulNBits.
