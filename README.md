# feln-webgpu

English question -> FELN query plan -> DuckDB SQL -> rows on a map, all inside one browser tab.
The language model is a fine-tuned [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) running on
WebGPU through transformers.js; the database is duckdb-wasm with the spatial extension; the map is the
ArcGIS Maps SDK for JavaScript 5.1.

```text
FELN.json + Layers.json ──prepare_data.py──> data/{train,val}.jsonl
                                    │
                            train.py (TRL, full FT, MPS)  ──> out/lfm2-350m-feln
                                    │
      evaluate.py (exact / structural / SQL executes on NorthSea.ddb)
                                    │
   onnx_swap.py: fine-tuned weights into onnx-community's LFM2-350M ONNX graph
   onnx_check.py: ONNX greedy decode == HF greedy decode
   onnx_quantize.py: fp16 -> 8-bit MatMulNBits          ──> web/model/onnx/
                                    │
   web/: index.html + app.js (transformers.js WebGPU, duckdb-wasm, feln_sql.js port, ArcGIS map)
```

## Data

Source folder: `~/Documents/ArcGIS/Projects/NorthSea` (`Layers.json` catalog, `FELN.json` 1,000
synthetic text/plan pairs from [feln](../feln), `NorthSea.ddb`).

- `data/`: 1,000 source rows plus 45 generated water-depth paraphrases ("wells deeper than 350 meters",
  "depth > 350"): FELN.json names that column in 5 rows only, always as "water depth", and the first model
  copied a user's bare "depth" into a non-existent column. Split 937 train / 108 val (source rows
  stratified by number of layers, paraphrases 1 in 5). System prompt = output contract + layer and
  column names; hints and value codes live in the weights.
- `web/data/`: `Wells`, `Discoveries`, `Pipelines` as parquet with WKB geometry (5.9 MB), `Layers.json`,
  the system prompt. `Wells_depth_stats` / `Wells_deep_400_stats` are in the catalog and training data
  but not in `NorthSea.ddb`, so the page reports "table not loaded" for them.

## Train and evaluate

```bash
uv run prepare_data.py
uv run train.py                       # ~20 min on M4 Max; opens a Terminal window with watch.py
uv run train.py --no-tui              # same, without the dashboard window
uv run evaluate.py data/val.jsonl --model out/lfm2-350m-feln
uv run evaluate.py data/val.jsonl --model LiquidAI/LFM2-350M   # untuned baseline
```

`train.py` writes `<out>/status.json` and `<out>/metrics.jsonl`; `uv run watch.py <out>` renders them.

Holdout, greedy decoding (3 epochs, lr 5e-5, full fine-tune):

| model | holdout | valid JSON | exact plan | structural | SQL compiles | OBJECTID Jaccard (executable rows) |
|---|---|---|---|---|---|---|
| LFM2-350M untuned | 99 | 0/99 | 0 | 0 | 0 | - |
| v1, source rows only | 99 | 99/99 | 92/99 (92.9%) | 0.993 | 99/99 | 0.965 (85) |
| v2, + depth paraphrases (`out/lfm2-350m-feln-v2`, shipped) | 108 | 108/108 | 103/108 (95.4%) | 0.996 | 108/108 | 0.989 (94) |

Remaining misses are code-table confusions (`discovery_type` Oil = 3 vs 4) and the `core_sample`
column, which is text `'YES'` where every sibling flag is an integer.

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
| v2 (shipped) | model_q8_fp16 | MatMulNBits 8-bit, block 128 | 443 MB | 85/90 | 89/90 | 0.40 s |

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

`app.js` loads `model/` (8-bit, 443 MB; `?model=model&dtype=fp16` for fp16) and shows a progress row per
downloaded file. The browser cache is keyed on `model/version.txt`, written by `onnx_quantize.py`, so a
re-export is never served from the previous version's cache (this bit once: the page kept the v1 weights), the three parquet tables, then for a
question: chat template -> greedy generate -> JSON -> `feln_sql.js` -> `SELECT ... WHERE OBJECTID IN (plan)`
-> table + graphics layer. `window.feln.ask(text)` is the same pipeline for tests.

Known limits:

- Phrasing outside the generator grammar can still misfire: "wells deeper than 500 meters and within 5
  kilometers of gas pipelines" folds the pipeline filter into the wells WHERE; the grammar says "The
  returned wells must be within 5 kilometers of gas pipelines". The page reports the invalid plan.
- Geometry is WGS84 degrees and `ST_DWithin` is fed metres, exactly as the Python `FELNToDuckDB`
  does; result sets match Python, but distances are not geodesic. Fix in `feln` first, then here.
- transformers.js needs `env.localModelPath` to be a relative path; an absolute URL silently skips
  the local lookup.
- CDN dependencies: `@huggingface/transformers`, `@duckdb/duckdb-wasm` (+ the spatial extension from
  extensions.duckdb.org), `js.arcgis.com/5.1`. Model and data are served locally.
