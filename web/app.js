// Text -> FELN (LFM2-350M fine-tune, transformers.js on WebGPU) -> DuckDB SQL (feln_sql.js)
// -> rows (duckdb-wasm + spatial). Everything runs in the browser; nothing leaves the tab.
import { AutoTokenizer, AutoModelForCausalLM, DynamicCache, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
import * as duckdb from "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.33.1-dev57.0/+esm";
import { compile } from "./feln_sql.js";
import { TABLES, loadSql } from "./db.js";

const $ = (id) => document.getElementById(id);
const status = (t, state = "") => { const el = $("status"); el.textContent = t; el.className = state; };
const loadingMsg = (t) => ($("loading").querySelector(".msg").textContent = t);
const MAX_ROWS = 200;

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./"; // must be a relative path: an absolute URL makes transformers.js skip the local lookup

// Browser cache is keyed on the export stamp written by onnx_quantize.py, so new weights under the
// same file names are fetched, not served from the previous version's cache. Old caches are dropped.
const version = (await (await fetch("./model/version.txt", { cache: "no-store" })).text()).trim();
env.cacheKey = `feln-${version}`;
for (const key of await caches.keys()) if (key !== env.cacheKey) caches.delete(key);

// ---- download progress: one row per file (model weights, tokenizer, data tables) ----
const MB = (n) => `${(n / 1e6).toFixed(1)} MB`;
function progressRow(name) {
  const box = $("loading");
  let bar = box.querySelector(`progress[data-name="${name}"]`);
  if (!bar) {
    box.appendChild(Object.assign(document.createElement("span"), { textContent: name }));
    bar = box.appendChild(Object.assign(document.createElement("progress"), { max: 1, value: 0 }));
    bar.dataset.name = name;
    bar.size = box.appendChild(Object.assign(document.createElement("span"), { textContent: "" }));
  }
  const report = (loaded, total) => {
    bar.value = total ? loaded / total : 0;
    bar.size.textContent = total ? `${MB(loaded)} / ${MB(total)}` : MB(loaded);
    bar.dataset.total = total;
  };
  report.done = () => {
    const total = Number(bar.dataset.total) || 0;
    if (total) report(total, total);
    else { bar.value = 1; bar.size.textContent = "cached"; } // no progress events: served from the browser cache
  };
  return report;
}

/** fetch() that reports bytes as they stream in. */
async function fetchBytes(url) {
  const res = await fetch(url);
  const total = Number(res.headers.get("content-length")) || 0;
  const report = progressRow(url.replace(/^\.\//, ""));
  const chunks = [];
  let loaded = 0;
  for (const reader = res.body.getReader(); ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    report(loaded, total);
  }
  report(loaded, total || loaded);
  report.done();
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function loadModel() {
  const device = navigator.gpu ? "webgpu" : "wasm";
  const params = new URLSearchParams(location.search);
  loadingMsg(`Downloading model (${device}) and data...`);
  const tokenizer = await AutoTokenizer.from_pretrained("model");
  const model = await AutoModelForCausalLM.from_pretrained("model", {
    device,
    // Default file: onnx/model_q8_fp16.onnx (8-bit MatMulNBits weights, fp16 activations): same holdout
    // accuracy as fp16 at 64% of the size; 4-bit lost ~7 points on 3-layer plans. ?model=model&dtype=fp16 loads the fp16 graph.
    dtype: params.get("dtype") ?? "fp16",
    model_file_name: params.get("model") ?? "model_q8",
    progress_callback: (p) => {
      if (p.status === "progress") progressRow(`model/${p.file}`)(p.loaded, p.total);
      if (p.status === "done") progressRow(`model/${p.file}`).done();
    },
  });
  return { tokenizer, model, device };
}

async function loadDb() {
  const bundle = await duckdb.selectBundle(duckdb.getJsDelivrBundles());
  const workerUrl = URL.createObjectURL(new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" }));
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), new Worker(workerUrl));
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  for (const t of TABLES) await db.registerFileBuffer(`${t}.parquet`, await fetchBytes(`./data/${t}.parquet`));
  const conn = await db.connect();
  for (const sql of loadSql((t) => `${t}.parquet`)) await conn.query(sql);
  return conn;
}

let tokenizer, model, device, conn, catalog, system;
try {
  [{ tokenizer, model, device }, conn, catalog, system] = await Promise.all([
    loadModel(),
    loadDb(),
    fetch("./data/Layers.json").then((r) => r.json()),
    fetch("./data/system_prompt.txt").then((r) => r.text()),
  ]);
} catch (err) {
  status("Failed to load", "error");
  $("loading").appendChild(Object.assign(document.createElement("div"), { className: "err", textContent: `${err.message}\n\nReload the page. If it persists, open the browser console.` }));
  throw err;
}
const loaded = new Set(TABLES);

// ---- map (ArcGIS Maps SDK for JavaScript 5.1, loaded by index.html) ----
const map = await (async () => {
  try {
    const el = $("map");
    await Promise.race([el.viewOnReady(), new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 15000))]);
    const [GraphicsLayer, Graphic] = await $arcgis.import(["@arcgis/core/layers/GraphicsLayer.js", "@arcgis/core/Graphic.js"]);
    const layer = new GraphicsLayer();
    el.map.add(layer);
    return { el, layer, Graphic };
  } catch (err) {
    console.warn("map unavailable:", err.message);
    return null;
  }
})();

const SYMBOLS = {
  point: { type: "simple-marker", size: 7, color: [230, 80, 30, 0.9], outline: { color: [255, 255, 255], width: 0.7 } },
  polyline: { type: "simple-line", color: [230, 80, 30, 0.9], width: 2.5 },
  polygon: { type: "simple-fill", color: [230, 80, 30, 0.35], outline: { color: [180, 50, 10], width: 1 } },
};

/** GeoJSON geometry (WGS84) -> ArcGIS geometry JSON. */
function toArcgis(g) {
  const sr = { wkid: 4326 };
  switch (g.type) {
    case "Point": return { type: "point", x: g.coordinates[0], y: g.coordinates[1], spatialReference: sr };
    case "MultiPoint": return { type: "multipoint", points: g.coordinates, spatialReference: sr };
    case "LineString": return { type: "polyline", paths: [g.coordinates], spatialReference: sr };
    case "MultiLineString": return { type: "polyline", paths: g.coordinates, spatialReference: sr };
    case "Polygon": return { type: "polygon", rings: g.coordinates, spatialReference: sr };
    case "MultiPolygon": return { type: "polygon", rings: g.coordinates.flat(), spatialReference: sr };
    default: return null;
  }
}

async function draw(rows) {
  if (!map) return;
  map.layer.removeAll();
  const graphics = [];
  for (const r of rows) {
    const geometry = r.__geojson && toArcgis(JSON.parse(r.__geojson));
    if (!geometry) continue;
    const attributes = Object.fromEntries(Object.entries(r).filter(([k]) => k !== "__geojson"));
    graphics.push(new map.Graphic({ geometry, symbol: SYMBOLS[geometry.type] ?? SYMBOLS.point, attributes, popupTemplate: { title: "{OBJECTID}", content: Object.keys(attributes).map((k) => `<b>${k}</b>: {${k}}`).join("<br>") } }));
  }
  map.layer.addMany(graphics);
  if (graphics.length) await map.el.goTo(graphics).catch(() => {});
}

function promptFor(text) {
  return tokenizer.apply_chat_template(
    [{ role: "system", content: system }, { role: "user", content: text }],
    { add_generation_prompt: true, return_dict: true }
  );
}

// ---- prefix cache ----
// The system prompt (the OKF bundle, ~10.9k tokens) is identical for every question, so its
// KV/conv state is computed once at load and reused: generate() sees a cache shorter than the
// prompt and only runs the question tokens through the model. The cached tensors live on the CPU,
// so DynamicCache.update() never disposes them and one copy serves every query.
const promptIds = (text) => promptFor(text).input_ids.tolist()[0];

async function buildPrefixCache() {
  const a = promptIds("a"), b = promptIds("Which wells are dry?");
  let n = 0;
  while (n < a.length && n < b.length && a[n] === b[n]) n++;
  const ids = a.slice(0, n);
  const out = await model.generate({
    input_ids: new Tensor("int64", BigInt64Array.from(ids), [1, n]),
    attention_mask: new Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]),
    max_new_tokens: 1, do_sample: false, past_key_values: new DynamicCache(), return_dict_in_generate: true,
  });
  const entries = {};
  for (const [key, t] of Object.entries(out.past_key_values)) entries[key] = new Tensor(t.type, await t.ort_tensor.getData(true), t.dims);
  return { ids, entries, tokens: n };
}

const prefix = new URLSearchParams(location.search).get("prefix") === "0" ? null : await buildPrefixCache();

async function textToFeln(text) {
  const inputs = promptFor(text);
  const ids = inputs.input_ids.tolist()[0];
  const cached = prefix && ids.length > prefix.ids.length && prefix.ids.every((v, i) => v === ids[i]);
  const opts = cached ? { past_key_values: new DynamicCache({ ...prefix.entries }), return_dict_in_generate: true } : {};
  const out = await model.generate({ ...inputs, max_new_tokens: 200, do_sample: false, ...opts });
  const seq = cached ? out.sequences : out;
  if (cached) await out.past_key_values.dispose();
  const raw = tokenizer.decode(seq.tolist()[0].slice(inputs.input_ids.dims[1]), { skip_special_tokens: true });
  try { return { raw, feln: JSON.parse(raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1)) }; }
  catch { return { raw, feln: null }; }
}

/** Quoted identifiers in each WHERE that are not columns of that layer. */
function unknownColumns(feln) {
  const bad = [];
  feln.layers.forEach((name, i) => {
    const cols = new Set((catalog.layers.find((l) => l.name === name)?.columns ?? []).map((c) => c.name.toLowerCase()));
    for (const [, id] of (feln.where[i] ?? "").matchAll(/"([^"]+)"/g)) if (!cols.has(id.toLowerCase())) bad.push(`${name}.${id}`);
  });
  return bad;
}

/** Full pipeline; also used by test/browser.mjs. */
async function ask(text) {
  const t0 = performance.now();
  const { raw, feln } = await textToFeln(text);
  const genMs = performance.now() - t0;
  if (!feln) return { feln, raw, genMs, rows: null, error: `the model returned no query plan: ${JSON.stringify(raw.slice(0, 120))}` };
  let sql;
  try { sql = compile(feln, catalog); } catch (err) { return { feln, raw, genMs, rows: null, error: err.message }; }
  const bad = unknownColumns(feln);
  if (bad.length) return { feln, raw, sql, genMs, rows: null, error: `the model used columns that do not exist: ${bad.join(", ")}. Try naming the field as in the catalog (e.g. "water depth").` };
  const missing = feln.layers.filter((l) => !loaded.has(catalog.layers.find((x) => x.name === l)?.table_name || l));
  if (missing.length) return { feln, sql, genMs, rows: null, error: `table not loaded in browser: ${missing.join(", ")}` };
  const layer = catalog.layers.find((x) => x.name === feln.layers[0]);
  const table = layer.table_name || layer.name;
  const res = await conn.query(`SELECT * EXCLUDE (geometry), ST_AsGeoJSON(geometry) AS __geojson FROM "${table}" WHERE OBJECTID IN (${sql}) ORDER BY OBJECTID LIMIT ${MAX_ROWS}`);
  const count = Number((await conn.query(`SELECT count(*) AS n FROM (${sql})`)).toArray()[0].n);
  return { feln, raw, sql, genMs, count, rows: res.toArray().map((r) => Object.fromEntries(Object.entries(r.toJSON()).map(([k, v]) => [k, typeof v === "bigint" ? Number(v) : v]))) };
}
/** OBJECTIDs of a compiled plan, sorted; used by the browser test. */
async function ids(sql) {
  return (await conn.query(`SELECT OBJECTID FROM (${sql}) ORDER BY OBJECTID`)).toArray().map((r) => Number(r.OBJECTID));
}
window.feln = { ask, ids, tokens: (text) => promptFor(text).input_ids.tolist()[0].map(Number) };

function render({ feln, sql, genMs, count, rows, error }) {
  $("feln").textContent = JSON.stringify(feln, null, 1);
  $("sql").textContent = sql ?? "";
  const table = $("rows");
  table.replaceChildren();
  map?.layer.removeAll();
  if (error) { $("rows-h").textContent = error; return; }
  $("rows-h").textContent = `${count} row${count === 1 ? "" : "s"}${rows.length < count ? ` (showing ${rows.length})` : ""} · model ${(genMs / 1000).toFixed(1)} s`;
  if (!rows.length) return;
  draw(rows);
  const cols = Object.keys(rows[0]).filter((c) => c !== "__geojson");
  const head = table.createTHead().insertRow();
  for (const c of cols) head.appendChild(Object.assign(document.createElement("th"), { textContent: c }));
  const body = table.createTBody();
  const fmt = (c, v) => (v == null ? "" : c.endsWith("_date") && typeof v === "number" ? new Date(v).toISOString().slice(0, 10) : v);
  for (const r of rows) { const tr = body.insertRow(); for (const c of cols) tr.insertCell().textContent = fmt(c, r[c]); }
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("go").disabled = true;
  status("Thinking");
  try { render(await ask($("q").value)); status(`Ready on ${device}${prefix ? ` (${prefix.tokens}-token prompt prefix cached)` : ""}`, "ready"); }
  catch (err) { status(`Error: ${err.message}`, "error"); }
  $("go").disabled = false;
});
status(`Ready on ${device}${prefix ? ` (${prefix.tokens}-token prompt prefix cached)` : ""}`, "ready");
$("loading-card").remove();
$("q").disabled = $("go").disabled = false;
$("q").focus();
