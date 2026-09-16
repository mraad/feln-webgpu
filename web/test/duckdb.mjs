// Runs the gold FELN plans of the val split through duckdb-wasm (node, blocking bundle)
// and compares OBJECTID sets with the Python/duckdb result in gold_ids.json.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { compile } from "../feln_sql.js";
import { TABLES, loadSql } from "../db.js";

const require = createRequire(import.meta.url);
const duckdb = require("@duckdb/duckdb-wasm/dist/duckdb-node-blocking.cjs");
const dist = path.dirname(require.resolve("@duckdb/duckdb-wasm"));
const dataDir = fileURLToPath(new URL("../data/", import.meta.url));
const catalog = JSON.parse(readFileSync(path.join(dataDir, "Layers.json")));
const gold = JSON.parse(readFileSync(new URL("./gold_ids.json", import.meta.url)));

const bundles = { eh: { mainModule: path.join(dist, "duckdb-eh.wasm"), mainWorker: "" } };
const db = await duckdb.createDuckDB(bundles, new duckdb.VoidLogger(), duckdb.NODE_RUNTIME);
await db.instantiate();
const conn = db.connect();
console.log("duckdb", db.getVersion());
for (const sql of loadSql((t) => path.join(dataDir, `${t}.parquet`))) conn.query(sql);
for (const t of TABLES) console.log(t, String(conn.query(`select count(*) n from "${t}"`).toArray()[0].n));

let bad = 0;
for (const { feln, ids } of gold) {
  const got = conn.query(compile(feln, catalog)).toArray().map((r) => Number(r.OBJECTID)).sort((a, b) => a - b);
  if (JSON.stringify(got) !== JSON.stringify(ids)) { bad++; if (bad <= 3) console.log("MISMATCH", JSON.stringify(feln), got.length, ids.length); }
}
console.log(`${gold.length - bad}/${gold.length} result sets identical`);
process.exit(bad ? 1 : 0);
