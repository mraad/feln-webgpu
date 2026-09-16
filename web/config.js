export default {
  title: "North Sea Query",
  modelUrl: "./model/", // Directory containing version.txt, tokenizer/config files and onnx/.
  dataUrl: "./data/", // Directory containing Layers.json, system_prompt.txt and Parquet tables.
  databaseUrl: null, // Optional prebuilt DuckDB file URL; null loads the Parquet tables instead.
};
