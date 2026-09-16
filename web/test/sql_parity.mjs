import { readFileSync } from "node:fs";
import { compile } from "../feln_sql.js";
const catalog = JSON.parse(readFileSync(new URL("../data/Layers.json", import.meta.url)));
const gold = JSON.parse(readFileSync(new URL("./gold_sql.json", import.meta.url)));
let bad = 0;
for (const { feln, sql } of gold) {
  const js = compile(feln, catalog);
  if (js !== sql) { bad++; if (bad <= 3) console.log("MISMATCH\n" + js + "\n---\n" + sql); }
}
console.log(`${gold.length - bad}/${gold.length} identical`);
process.exit(bad ? 1 : 0);
