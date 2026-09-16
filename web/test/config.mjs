// Browser configuration smoke test with small, mocked model/database downloads.
// Run: node test/config.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const browser = await chromium.launch();
try {
  for (const databaseUrl of [null, "https://assets.example/database.duckdb"]) {
    const page = await browser.newPage();
    const requests = [];
    const config = { title: "Configured Query", modelUrl: "https://assets.example/weights", dataUrl: "./custom-data", databaseUrl };
    await page.route("**/*", async (route) => {
      const url = route.request().url();
      requests.push(url);
      let body, contentType = "text/javascript";
      if (url.includes("@huggingface/transformers")) {
        body = `
          export const env = {};
          export class DynamicCache {};
          export class Tensor {};
          export const AutoTokenizer = { from_pretrained: async () => {
            if (env.allowLocalModels || !env.allowRemoteModels || env.remotePathTemplate !== '') throw new Error('Wrong model routing');
            await fetch(env.remoteHost + 'tokenizer.json');
            return { apply_chat_template: () => ({ input_ids: { tolist: () => [[1n, 2n]] } }) };
          }};
          export const AutoModelForCausalLM = { from_pretrained: async () => {
            await fetch(env.remoteHost + 'onnx/model_q8_fp16.onnx');
            await new Promise(resolve => { window.finishModel = resolve; });
            return { generate: async () => {
              await new Promise(resolve => { window.finishWarmup = resolve; });
              return { past_key_values: {} };
            }};
          }};
        `;
      } else if (url.includes("@duckdb/duckdb-wasm")) {
        body = `
          export const getJsDelivrBundles = () => ({}), selectBundle = async () => ({});
          export const DuckDBAccessMode = { READ_ONLY: 1 };
          export class VoidLogger {};
          export class AsyncDuckDB {
            async instantiate() {}
            async registerFileBuffer() {}
            async open(options) { window.databaseOptions = options; }
            async connect() {
              await new Promise(resolve => { window.finishDb = resolve; });
              return { query: async () => {} };
            }
          }
        `;
      } else if (url.startsWith("https://js.arcgis.com/")) {
        body = "";
      } else if (url.includes("/custom-data/Layers.json")) {
        body = '{"layers":[]}'; contentType = "application/json";
      } else if (url.endsWith("/config.js")) {
        body = `export default ${JSON.stringify(config)}`;
      } else if (/\/(app\.js|db\.js|feln_sql\.js|index\.html)(\?|$)/.test(url)) {
        const file = new URL(url).pathname.split("/").pop();
        body = readFileSync(new URL(`../${file}`, import.meta.url), "utf8");
        if (file.endsWith("html")) contentType = "text/html";
      } else {
        body = "fixture"; contentType = "text/plain";
      }
      await route.fulfill({ body, contentType });
    });
    await page.goto("http://localhost:8781/index.html");
    await page.waitForFunction(() => window.finishModel && window.finishDb);
    assert.equal(await page.title(), config.title);
    assert.equal(await page.textContent("#app-title"), config.title);
    await page.keyboard.press("Escape");
    assert.equal(await page.$eval("#loading-card", el => el.matches(":modal")), true);
    await page.evaluate(() => window.finishModel());
    assert.equal(await page.$eval("#loading-card", el => el.matches(":modal")), true);
    assert.equal(await page.$eval("#q", el => el.disabled), true);
    await page.evaluate(() => window.finishDb());
    await page.waitForFunction(() => window.finishWarmup);
    await page.keyboard.press("Escape");
    assert.equal(await page.$eval("#loading-card", el => el.matches(":modal")), true);
    assert.equal(await page.$eval("#q", el => el.disabled), true);
    await page.evaluate(() => window.finishWarmup());
    await page.waitForFunction(() => window.feln);
    assert.equal(await page.$eval("#loading-card", el => el.open), false);
    assert.equal(await page.$eval("#q", el => el.disabled), false);
    assert.equal(await page.$eval("#go", el => el.disabled), false);
    assert.equal(await page.$eval("#q", el => el === document.activeElement), true);
    for (const file of ["version.txt", "tokenizer.json", "onnx/model_q8_fp16.onnx"]) {
      assert(requests.includes(`https://assets.example/weights/${file}`));
    }
    assert(requests.includes("http://localhost:8781/custom-data/Layers.json"));
    assert(requests.includes("http://localhost:8781/custom-data/system_prompt.txt"));
    if (databaseUrl) {
      assert(requests.includes(databaseUrl));
      assert(!requests.some(url => url.endsWith(".parquet")));
      assert.deepEqual(await page.evaluate(() => window.databaseOptions), { path: "database.duckdb", accessMode: 1 });
    } else {
      for (const table of ["Wells", "Discoveries", "Pipelines"]) assert(requests.includes(`http://localhost:8781/custom-data/${table}.parquet`));
    }
    await page.close();
  }
  console.log("PASS: configured title, model/data URLs, database URL, and popup lifecycle");
} finally {
  await browser.close();
}
