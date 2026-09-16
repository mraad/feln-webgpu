// End-to-end in a real Chromium: loads index.html (served at BASE), waits for the model and
// DuckDB, then asks the holdout questions through window.feln.ask and scores FELN exact
// match plus OBJECTID-set match against gold_ids.json.
//   node test/browser.mjs [n]      (BASE=http://127.0.0.1:8765 by default)
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://127.0.0.1:8765";
const n = Number(process.argv[2] ?? 20);
const gold = JSON.parse(readFileSync(new URL("./gold_ids.json", import.meta.url)));
const val = readFileSync(new URL("../../data/val.jsonl", import.meta.url), "utf8").trim().split("\n").map((l) => JSON.parse(l));
const textOf = new Map(val.map((r) => [r.completion[0].content, r.prompt[1].content]));
const cases = gold.map((g) => ({ ...g, text: textOf.get(JSON.stringify(g.feln)) })).filter((c) => c.text).slice(0, n);

const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,SkiaGraphite", "--use-angle=metal", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
page.on("console", (m) => m.type() === "error" && console.log("console:", m.text()));
page.on("pageerror", (e) => console.log("pageerror:", e.message));
const query = new URLSearchParams(Object.fromEntries(["model", "dtype"].filter((k) => process.env[k.toUpperCase()]).map((k) => [k, process.env[k.toUpperCase()]])));
await page.goto(`${BASE}/index.html?${query}`);
await page.waitForFunction(() => window.feln?.ask, null, { timeout: 600_000 });
console.log("page ready:", await page.textContent("#status"));

if (process.env.TOKENS) console.log("tokens:", JSON.stringify(await page.evaluate((t) => window.feln.tokens(t), cases[0].text)));
let exact = 0, sets = 0, ms = 0;
for (const c of cases) {
  let r;
  try { r = await page.evaluate((t) => window.feln.ask(t), c.text); }
  catch (e) { console.log("ERR ", c.text, "\n  ", e.message.split("\n")[0]); continue; }
  ms += r.genMs;
  if (r.error) { console.log("ERR ", c.text, "\n  raw", r.raw, "\n  ", r.error); continue; }
  if (process.env.SCREENSHOT && exact === 0) {
    await page.fill("#q", c.text); await page.click("#go");
    await page.waitForFunction(() => document.getElementById("rows").rows.length > 0, null, { timeout: 60000 });
    await page.waitForTimeout(4000);
    await page.screenshot({ path: process.env.SCREENSHOT, fullPage: true });
  }
  const same = JSON.stringify(r.feln) === JSON.stringify(c.feln);
  exact += same;
  const ids = r.rows === null ? null : (await page.evaluate((sql) => window.feln.ids(sql), r.sql));
  const setOk = ids !== null && JSON.stringify(ids) === JSON.stringify(c.ids);
  sets += setOk;
  if (!same || !setOk) console.log(same ? "SET " : "FELN", c.text, "\n  got", JSON.stringify(r.feln), "\n  gold", JSON.stringify(c.feln), r.error ?? "");
}
console.log(`FELN exact ${exact}/${cases.length}, result sets ${sets}/${cases.length}, mean generate ${(ms / cases.length / 1000).toFixed(2)} s`);
await browser.close();
process.exit(exact === cases.length && sets === cases.length ? 0 : 1);
