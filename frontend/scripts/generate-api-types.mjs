import fs from "node:fs";
import path from "node:path";

const input = path.resolve("contracts/openapi.json");
const output = path.resolve("types/generated/openapi.ts");
const doc = JSON.parse(fs.readFileSync(input, "utf8").replace(/^\uFEFF/, ""));
const schemas = doc.components?.schemas ?? {};

function refName(ref) {
  return ref.split("/").at(-1);
}

function toTs(schema, indent = "") {
  if (!schema || Object.keys(schema).length === 0) return "unknown";
  if (schema.$ref) return refName(schema.$ref);
  if (schema.enum) return schema.enum.map((v) => JSON.stringify(v)).join(" | ");
  if (schema.anyOf) return schema.anyOf.map((s) => toTs(s, indent)).join(" | ");
  if (schema.type === "array") return `Array<${toTs(schema.items ?? {}, indent)}>`;
  if (schema.type === "string") return "string";
  if (schema.type === "number" || schema.type === "integer") return "number";
  if (schema.type === "boolean") return "boolean";
  if (schema.type === "null") return "null";
  if (schema.type === "object" || schema.properties || schema.additionalProperties) {
    if (!schema.properties && schema.additionalProperties) {
      return `Record<string, ${schema.additionalProperties === true ? "unknown" : toTs(schema.additionalProperties, indent)}>`;
    }
    const required = new Set(schema.required ?? []);
    const fields = Object.entries(schema.properties ?? {}).map(([key, value]) => {
      const optional = required.has(key) ? "" : "?";
      return `${indent}  ${JSON.stringify(key)}${optional}: ${toTs(value, indent + "  ")};`;
    });
    if (schema.additionalProperties && schema.additionalProperties !== false) {
      fields.push(`${indent}  [key: string]: ${schema.additionalProperties === true ? "unknown" : toTs(schema.additionalProperties, indent + "  ")};`);
    }
    return `{\n${fields.join("\n")}\n${indent}}`;
  }
  return "unknown";
}

const lines = [
  "/* eslint-disable */",
  "/**",
  " * AUTO-GENERATED from contracts/openapi.json.",
  " * Do not edit by hand. Run `npm run generate:api` after replacing the contract artifact.",
  " */",
  "",
];

for (const [name, schema] of Object.entries(schemas)) {
  lines.push(`export type ${name} = ${toTs(schema)};`, "");
}

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, lines.join("\n"));
console.log(`Generated ${Object.keys(schemas).length} schemas -> ${path.relative(process.cwd(), output)}`);
