# feln-webgpu: text -> FELN -> DuckDB, fully in browser

Source data: ~/Documents/ArcGIS/Projects/NorthSea/{Layers.json,FELN.json,NorthSea.ddb}
Prior work: NorthSea/feln-ft (Qwen2.5-Coder-7B MLX LoRA; schema-in-weights, names-only prompt worked best)

## Plan

- [x] 1. Project setup: `uv init`, deps torch (MPS) / transformers / trl / datasets / onnx / onnxruntime / feln (local)
- [x] 2. Data: FELN.json -> chat JSONL (system = contract + layer names; assistant = compact meta JSON); 900/100 split stratified by arity; optional augment via `feln generate`
- [x] 3. Baseline eval of untuned base model (exact match + FELNCompare.structural + SQL compiles + OBJECTID Jaccard on NorthSea.ddb)
- [x] 4. Fine-tune small model (full FT, completion-only loss, MPS) -> HF safetensors
- [x] 5. Eval tuned model on holdout; iterate once if < ~90% exact
- [x] 6. Export to ONNX (q4f16 + fp16), parity check with onnxruntime in Python
- [x] 7. Browser data: DDB tables -> parquet (WKB geometry) or direct .ddb attach in duckdb-wasm + spatial ext; pick whichever loads
- [x] 8. Browser app: index.html + ES modules (transformers.js WebGPU -> FELN JSON -> JS port of FELNToDuckDB -> duckdb-wasm) showing rows
- [x] 9. End-to-end check in browser on holdout questions; write README

## Decisions
- Base model: LFM2-350M. Stats tables: kept in training, skipped in browser. Data: 1000 as-is.
- Map: ArcGIS Maps SDK for JavaScript 5.1 (user request, 2026-09-16).

## Old open decisions
- Base model: LFM2-350M (recommended; onnx-community has ONNX + transformers.js support) vs Qwen2.5-Coder-0.5B vs LFM2-700M
- Wells_depth_stats / Wells_deep_400_stats: in Layers.json + 138 FELN samples, NOT in NorthSea.ddb
