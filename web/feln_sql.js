// JS port of feln.sql.FELNToDuckDB (+ sanitize, units, parse_relation). Output is byte-identical
// to the Python compiler with default options (objectid=OBJECTID, geometry=geometry, no st_as);
// test/sql_parity.mjs checks that against every FELN.json sample.

const IDENT_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const DANGEROUS_RE = /\b(DROP|CREATE|ALTER|TRUNCATE|INSERT|UPDATE|DELETE|MERGE|EXEC|EXECUTE|GRANT|REVOKE|ATTACH|DETACH|COPY|EXPORT|IMPORT|UNION\s+ALL|UNION\s+SELECT)\b/i;
const METERS_PER = { m: 1, meter: 1, meters: 1, km: 1000, kilometer: 1000, kilometers: 1000, ft: 0.3048, foot: 0.3048, feet: 0.3048, mi: 1609.34, mile: 1609.34, miles: 1609.34, yd: 0.9144, yard: 0.9144, yards: 0.9144 };
const KINDS = { within: "within", inside: "inside", contains: "contains", intersects: "intersects", withindistance: "withinDistance", notwithindistance: "notWithinDistance", none: "none" };

function stripStringLiterals(clause) {
  const out = clause.split("");
  for (let i = 0; i < clause.length; ) {
    const ch = clause[i];
    if (ch === "'" || ch === '"') {
      out[i++] = " ";
      while (i < clause.length) {
        if (clause[i] === ch) {
          if (clause[i + 1] === ch) { out[i] = " "; out[i + 1] = " "; i += 2; }
          else { out[i++] = " "; break; }
        } else out[i++] = " ";
      }
    } else i++;
  }
  return out.join("");
}

export function sanitizeIdentifier(name) {
  if (!name || !IDENT_RE.test(name)) throw new Error(`Invalid SQL identifier: ${JSON.stringify(name)}`);
  return name;
}

export function sanitizeWhere(where) {
  const unquoted = stripStringLiterals(where);
  if (unquoted.includes(";")) throw new Error(`WHERE clause contains prohibited semicolon: ${where}`);
  if (unquoted.includes("--") || unquoted.includes("/*")) throw new Error(`WHERE clause contains prohibited comment sequence: ${where}`);
  if (DANGEROUS_RE.test(unquoted)) throw new Error(`WHERE clause contains prohibited keyword: ${where}`);
  return where;
}

export function tableIdent(layer) {
  return sanitizeIdentifier(layer.table_name || layer.name.replaceAll(" ", "_"));
}

export function parseRelation(rel) {
  const parts = rel.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return { kind: "none" };
  const kind = KINDS[parts[0].toLowerCase()];
  if (!kind) throw new Error(`Unknown relation kind: '${parts[0]}'`);
  if (kind === "withinDistance" || kind === "notWithinDistance") {
    if (parts.length !== 3) throw new Error(`${kind} relation must be '${kind} <value> <unit>', got: '${rel}'`);
    const distance = Number(parts[1]);
    if (!Number.isFinite(distance)) throw new Error(`Invalid distance value: '${parts[1]}'`);
    return { kind, distance, unit: parts[2].toLowerCase() };
  }
  return { kind };
}

// Python repr of a float: 20000.0 not 20000. Mirrors f"{float}" in the CTE text.
function pyFloat(x) {
  return Number.isInteger(x) ? `${x}.0` : `${x}`;
}

function distanceM(rel) {
  const f = METERS_PER[rel.unit.trim().toLowerCase()];
  return pyFloat(f === undefined ? rel.distance : rel.distance * f);
}

function onClause(rel, a, b) {
  switch (rel.kind) {
    case "contains": return `ST_Contains(${a}, ${b})`;
    case "intersects": return `ST_Intersects(${a}, ${b})`;
    case "within": case "inside": return `ST_Within(${a}, ${b})`;
    case "withinDistance": return `ST_DWithin(${a}, ${b}, ${distanceM(rel)})`;
    default: throw new Error(`Unknown relation kind: ${rel.kind}`);
  }
}

function singleLayer(layer, where) {
  const w = where ? `\nWHERE ${sanitizeWhere(where)}` : "";
  return `SELECT\nL.OBJECTID\nFROM ${tableIdent(layer)} L${w}`;
}

/** layers: catalog Layer objects ({name, table_name}); wheres: string[]; relations: string[] */
export function felnToSql(layers, wheres, relations) {
  const n = layers.length;
  if (!n) throw new Error("layers must contain at least one layer.");
  if (wheres.length !== n) throw new Error(`wheres must have ${n} entries, got ${wheres.length}.`);
  if (relations.length !== n - 1) throw new Error(`relations must have ${n - 1} entries for ${n} layers, got ${relations.length}.`);
  const rels = relations.map(parseRelation);
  if (n === 1) return singleLayer(layers[0], wheres[0]);

  const ctes = layers.map((layer, i) => {
    const w = wheres[i] ? ` WHERE ${sanitizeWhere(wheres[i])}` : "";
    return `L${i} AS (\n    SELECT OBJECTID, geometry\n    FROM ${tableIdent(layer)}${w}\n)`;
  });
  const joins = [];
  rels.forEach((rel, i) => {
    if (rel.kind === "none") {
      if (wheres[i + 1]) throw new Error(`Relation ${i} is a no-op but layer '${layers[i + 1].name}' has a non-empty WHERE clause ('${wheres[i + 1]}'). A no-op relation skips the spatial join, so the secondary layer's filter would be silently ignored.`);
      return;
    }
    if (rel.kind === "notWithinDistance") {
      ctes.push(`J${i} AS (\n    SELECT A.OBJECTID\n    FROM L0 A\n    WHERE NOT EXISTS (\n        SELECT 1 FROM L${i + 1} B\n        WHERE ST_DWithin(A.geometry, B.geometry, ${distanceM(rel)})\n    )\n)`);
    } else {
      ctes.push(`J${i} AS (\n    SELECT DISTINCT A.OBJECTID\n    FROM L0 A\n    JOIN L${i + 1} B\n    ON ${onClause(rel, "A.geometry", "B.geometry")}\n)`);
    }
    joins.push(i);
  });
  if (!joins.length) return singleLayer(layers[0], wheres[0]);
  const joinClauses = joins.map((i) => `JOIN J${i} M${i} ON L.OBJECTID = M${i}.OBJECTID`).join("\n");
  return `WITH ${ctes.join(",\n")}\nSELECT\n    L.OBJECTID\nFROM ${tableIdent(layers[0])} L\n${joinClauses}`;
}

/** Resolve a FELN {layers, where, relations} against a Layers.json catalog. */
export function compile(feln, catalog) {
  const layers = feln.layers.map((name) => {
    const layer = catalog.layers.find((l) => l.name === name);
    if (!layer) throw new Error(`unknown layer '${name}'`);
    return layer;
  });
  return felnToSql(layers, feln.where, feln.relations);
}
