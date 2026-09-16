# feln-webgpu

English question -> FELN query plan -> DuckDB SQL -> rows on a map, all inside one browser tab.
The language model is a fine-tuned [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) running on
WebGPU through transformers.js; the database is duckdb-wasm with the spatial extension; the map is the
ArcGIS Maps SDK for JavaScript 5.1.

```text
NorthSea.aprx + .gdb ──regen_northsea.py──> Layers.json, okf/, FELN.json (humanized) + validation
                                    │
FELN.json + Layers.json + okf/ ──prepare_data.py --okf──> data/{train,val}.jsonl, system_prompt.txt
                                    │
                            train.py (TRL, full FT; MPS locally or CUDA on the RTX box)  ──> out/<run>
                                    │
      evaluate.py (exact / structural / SQL executes on NorthSea.ddb)
                                    │
   onnx_swap.py: fine-tuned weights into onnx-community's LFM2-350M ONNX graph
   onnx_check.py: ONNX greedy decode == HF greedy decode
   onnx_quantize.py: fp16 -> 8-bit MatMulNBits          ──> web/model/onnx/
                                    │
   web/: index.html + app.js (transformers.js WebGPU, duckdb-wasm, feln_sql.js port, ArcGIS map)
```

## Regenerating the NorthSea artifacts

```bash
uv run regen_northsea.py            # Layers.json, okf/, FELN.json (humanized), validation JSON
uv run regen_northsea.py --no-humanize
```

`regen_northsea.py` backs up the current files to `NorthSea/backups/regen-<utc>/`, writes
`Layers.json` with `layers-json` (GDAL reader over `NorthSea.aprx`, `--use-ilike`), keeps the column
visibility of the previous catalog (Pro hid `symbol`, `casing_lot`, ... in the map) and reapplies
feln-liquid's alias refinements, writes the OKF bundle with `layers-okf`, generates 1,000 FELN plans
(seed 20260916, deduplicated on the normalized WHERE, audited with feln-liquid's `audit`), then rewrites
each grammar text through `claude -p` (Claude Opus 5, settings and tools disabled) so it reads like a
person typed it. A rewrite is kept only if every quoted literal, number and distance unit of the grammar
text survives and the text is unique; the grammar text stays on the record as `source_text`.
The two Pro-toolbox statistics tables are not produced by `layers-json` and were dropped (2026-09-16).

## Data

Source folder: `~/Documents/ArcGIS/Projects/NorthSea` (`Layers.json` catalog, `FELN.json` 1,000
synthetic text/plan pairs from [feln](../feln), `NorthSea.ddb`).

- `data/` (phase 2): 1,000 humanized rows + their grammar wording + 45 water-depth paraphrases + phase
  templates + conjoined-constraint rewrites of the training grammar rows. A plan's wordings all land on
  the same side of the 10% split. `prepare_data.py --okf NorthSea/okf` puts the OKF bundle verbatim in
  the system prompt (~10.9k tokens per row; `train.py --max-length 12288 --batch 4`).
- `data/` (phase 1): 1,000 source rows plus 45 generated water-depth paraphrases ("wells deeper than 350 meters",
  "depth > 350"): FELN.json names that column in 5 rows only, always as "water depth", and the first model
  copied a user's bare "depth" into a non-existent column. Split 937 train / 108 val (source rows
  stratified by number of layers, paraphrases 1 in 5). System prompt = output contract + layer and
  column names; hints and value codes live in the weights.
- `web/data/`: `Wells`, `Discoveries`, `Pipelines` as parquet with WKB geometry (5.9 MB), `Layers.json`,
  the system prompt (the same OKF prompt the model was trained on).
- Plans use the generator's cast form (`PipelinesType = cast(1 as SMALLINT)`, bare identifiers), which
  the OKF hints prescribe. Phase 1 trained on feln-liquid's normalized form (`"pipelinestype" = 1`).
  DuckDB runs both; the browser gold sets were rebuilt from the cast form.

## Train and evaluate

```bash
uv run prepare_data.py --okf ~/Documents/ArcGIS/Projects/NorthSea/okf
uv run train.py --out out/lfm2-350m-feln-v4 --batch 4 --max-length 12288      # opens a Terminal window with watch.py
uv run train.py --no-tui ...                                                  # without the dashboard window
uv run train.py --base LiquidAI/LFM2.5-1.2B-Instruct --lr 2e-5 --batch 4 --max-length 12288 --out out/lfm25-1.2b-feln-v4
uv run evaluate.py data/val.jsonl --model out/lfm2-350m-feln-v4
uv run evaluate.py data/val.jsonl --model LiquidAI/LFM2-350M   # untuned baseline
```

`train.py` writes `<out>/status.json` and `<out>/metrics.jsonl`; `uv run watch.py <out>` renders them
(eval loss appears on the epoch-end rows only). On the RTX box the same commands run from
`~/feln-webgpu` (uv venv, torch cu130); copy `data/*.jsonl` + `system_prompt.txt` there, run with
`--no-tui`, copy `out/<run>` back and export here. A 350M run with the OKF prompt takes 31 min on one
RTX PRO 6000 (36 s with the short phase-1 prompt); the 1.2B run took 71 min.

Holdout, greedy decoding (3 epochs, lr 5e-5, full fine-tune):

| model | holdout | valid JSON | exact plan | structural | SQL compiles | OBJECTID Jaccard (executable rows) |
|---|---|---|---|---|---|---|
| LFM2-350M untuned | 99 | 0/99 | 0 | 0 | 0 | - |
| v1, source rows only | 99 | 99/99 | 92/99 (92.9%) | 0.993 | 99/99 | 0.965 (85) |
| v2, + depth paraphrases | 108 | 108/108 | 103/108 (95.4%) | 0.996 | 108/108 | 0.989 (94) |
| v3, + conjoined constraints and phase adjectives, M4 Max (11 min) | 175 | 175/175 | 168/175 (96.0%) | 0.995 | 175/175 | 0.975 (161) |
| v3, same data on an RTX PRO 6000 (36 s), `out/lfm2-350m-feln-v3-rtx` | 175 | 175/175 | 169/175 (96.6%) | 0.995 | 175/175 | 0.983 (161) |
| v4, phase-2 data (humanized + grammar), OKF system prompt, RTX (31 min), shipped as `out/lfm2-350m-feln-v4` | 210 | 210/210 | 203/210 (96.7%) | 0.997 | 210/210 | 0.990 (210) |
| v4 on LiquidAI/LFM2.5-1.2B-Instruct (feln-liquid's base), same data, RTX GPU 1 (71 min, lr 2e-5), `out/lfm25-1.2b-feln-v4`, not exported | 210 | 210/210 | 204/210 (97.1%) | 0.999 | 210/210 | 0.995 (210) |

Both v4 models miss the same questions, so the gap is in the data, not the model: humanized wordings
that outrun the schema vocabulary ("a blank content field" -> invented column `content_field`; "a
source other than Department of Energy & Climate Change" -> label instead of the stored code `DECC`;
"pipelines of type Unknown whose to facility is 'KOLLSNES' or 'TEESSIDE'" -> OR grouping), plus two
three-layer questions. The 350M alone also misses "Unknown pipelines outside Denmark" (`=` for `<>`).
The 1.2B gains 0.5 points for a ~1.5 GB 8-bit download and about three times the WebGPU cost per
token, so the 350M stays in the app.

## Export to the browser

optimum-onnx has no LFM2 exporter, so the fine-tuned weights are written into the initializers of
onnx-community's `LFM2-350M-ONNX` graph (same architecture); the mapping is derived from the base
weights by value, so no exporter naming is assumed.

```bash
uv run onnx_swap.py --base-onnx scratch/model_fp16.onnx --model out/lfm2-350m-feln --out web/model/onnx/model_fp16.onnx
uv run onnx_check.py --onnx web/model/onnx/model_fp16.onnx --model out/lfm2-350m-feln --n 5    # 5/5 identical
uv run onnx_quantize.py web/model/onnx/model_fp16.onnx web/model/onnx/model_q8_fp16.onnx
```

Browser (headless Chromium, WebGPU, holdout questions whose plans execute on the loaded tables).
Quantization sweep on v1 (81 questions), then v2 with the chosen format (90 questions):

| model | onnx file | weights | size | exact plan | result sets | generate |
|---|---|---|---|---|---|---|
| v1 | model_fp16 | fp16 | 692 MB | 74/81 | 77/81 | 0.68 s |
| v1 | model_q8_fp16 | MatMulNBits 8-bit, block 128 | 443 MB | 74/81 | 77/81 | 0.41 s |
| v1 | q4f16, block 32 symmetric | MatMulNBits 4-bit | 298 MB | 68/81 | 72/81 | 0.23 s |
| v1 | q4f16, block 32 / 128 asymmetric | MatMulNBits 4-bit | 300-317 MB | 71/81 | 74-75/81 | 0.25-0.40 s |
| v2 | model_q8_fp16 | MatMulNBits 8-bit, block 128 | 443 MB | 85/90 | 89/90 | 0.40 s |
| v3 RTX | model_q8_fp16 | MatMulNBits 8-bit, block 128 | 443 MB | 155/161 | 157/161 | 0.40 s |
| v4 (shipped), OKF prompt, prefix cached | model_q8_fp16 | MatMulNBits 8-bit, block 128 | 443 MB | 203/210 | 208/210 | 0.46 s |

`onnx.save` appends to an existing external-data file; both export scripts unlink it first.

`scratch/model_fp16.onnx{,_data}` is `onnx-community/LFM2-350M-ONNX/onnx/model_fp16.onnx`.
`web/model/*.json` (config, tokenizer) come from the same repo; the tokenizer is unchanged by
fine-tuning (checked: identical encodings on the val split).

## Web app

```bash
cd web && npm install
npm test                 # feln_sql.js == Python compiler on all 1000 plans; duckdb-wasm result sets == Python on val
npm run serve            # http://127.0.0.1:8765
npm run test:browser     # Playwright + headless Chromium with WebGPU, end to end on the val questions
```

`app.js` loads `model/` (8-bit, 443 MB; `?model=model&dtype=fp16` for fp16) and the three parquet
tables, with a progress row per downloaded file; the loading card is removed when ready and shows the
error if a download fails. The browser cache is keyed on `model/version.txt`, written by
`onnx_quantize.py`, so a re-export is never served from the previous version's cache (this bit once:
the page kept the v1 weights). Per question: chat template -> greedy generate -> JSON -> column names
checked against the catalog -> `feln_sql.js` -> `SELECT ... WHERE OBJECTID IN (plan)` -> table +
graphics layer on the dark-gray basemap. `window.feln.ask(text)` is the same pipeline for tests.

With the OKF system prompt the page prefills the ~10.8k-token prefix once at load (adds ~7 s) and
reuses its KV/conv cache for every question (`buildPrefixCache` in `app.js`; `?prefix=0` disables it).
Measured on the same weights: 7.0 s per question without the cache, 0.7-1.3 s with it.

Known limits:

- The OKF docs embed absolute `file:///Users/...` paths in `resource:`/`sources:`, so the shipped
  system prompt contains the author's home path. Harmless for the model; strip it if that matters.

- v3 data adds three paraphrase families on top of FELN.json: water-depth wording, "in-service /
  decommissioned / abandoned pipelines" for `current_phase`, and constraints joined into one sentence
  ("wells that are within 4 miles of ... and contain ..."). Other wording outside the grammar can still
  misfire; the page reports an invalid plan instead of running it.
- Training elsewhere: `train.py --no-tui` on a CUDA box with the same `data/*.jsonl` and pinned
  `transformers`/`trl` gives an equivalent checkpoint (RTX PRO 6000: 36 s for the full run); copy
  `out/<name>` back and run the export steps here.
- Geometry is WGS84 degrees and `ST_DWithin` is fed metres, exactly as the Python `FELNToDuckDB`
  does; result sets match Python, but distances are not geodesic. Fix in `feln` first, then here.
- transformers.js needs `env.localModelPath` to be a relative path; an absolute URL silently skips
  the local lookup.
- CDN dependencies: `@huggingface/transformers`, `@duckdb/duckdb-wasm` (+ the spatial extension from
  extensions.duckdb.org), `js.arcgis.com/5.1`. Model and data are served locally.
