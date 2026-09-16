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

- `data/`: 901 train / 99 val (stratified by number of layers), system prompt = output contract +
  layer and column names. Hints and value codes live in the weights.
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

Holdout (99 questions, greedy):

| model | valid JSON | exact plan | structural | SQL compiles | OBJECTID Jaccard (85 executable) |
|---|---|---|---|---|---|
| LFM2-350M untuned | 0/99 | 0 | 0 | 0 | - |
| fine-tuned, 3 epochs, lr 5e-5, full FT | 99/99 | 92/99 (92.9%) | 0.993 | 99/99 | 0.965 |

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

Browser (headless Chromium, WebGPU, the 81 holdout questions whose plans execute on the loaded tables):

| onnx file | weights | size | exact plan | result sets | generate |
|---|---|---|---|---|---|
| model_fp16 | fp16 | 692 MB | 74/81 | 77/81 | 0.68 s |
| model_q8_fp16 (default) | MatMulNBits 8-bit, block 128 | 443 MB | 74/81 | 77/81 | 0.41 s |
| q4f16, block 32 symmetric | MatMulNBits 4-bit | 298 MB | 68/81 | 72/81 | 0.23 s |
| q4f16, block 32 / 128 asymmetric | MatMulNBits 4-bit | 300-317 MB | 71/81 | 74-75/81 | 0.25-0.40 s |

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

`app.js` loads `model/` (8-bit, 443 MB, cached by the browser; `?model=model&dtype=fp16` for fp16), the three parquet tables, then for a
question: chat template -> greedy generate -> JSON -> `feln_sql.js` -> `SELECT ... WHERE OBJECTID IN (plan)`
-> table + graphics layer. `window.feln.ask(text)` is the same pipeline for tests.

Known limits:

- Geometry is WGS84 degrees and `ST_DWithin` is fed metres, exactly as the Python `FELNToDuckDB`
  does; result sets match Python, but distances are not geodesic. Fix in `feln` first, then here.
- transformers.js needs `env.localModelPath` to be a relative path; an absolute URL silently skips
  the local lookup.
- CDN dependencies: `@huggingface/transformers`, `@duckdb/duckdb-wasm` (+ the spatial extension from
  extensions.duckdb.org), `js.arcgis.com/5.1`. Model and data are served locally.
