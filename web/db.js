// DuckDB-wasm setup shared by the browser app and the node test: spatial extension +
// the three NorthSea tables loaded from parquet (geometry stored as WKB).
export const TABLES = ["Wells", "Discoveries", "Pipelines"];

/** SQL statements that install spatial and create the tables; urlFor(table) -> parquet location. */
export function loadSql(urlFor) {
  return [
    "INSTALL spatial",
    "LOAD spatial",
    ...TABLES.map(
      (t) => `CREATE TABLE "${t}" AS SELECT * EXCLUDE (geometry), ST_GeomFromWKB(geometry) AS geometry FROM read_parquet('${urlFor(t)}')`
    ),
  ];
}
