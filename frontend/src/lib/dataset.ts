import dataset from "../../../corfu_dataset.json";
import type { POI } from "./types";

interface DatasetFile {
  dataset: string;
  version: string;
  timezone: string;
  pois: POI[];
}

const source = dataset as DatasetFile;

function isPOI(value: unknown): value is POI {
  if (!value || typeof value !== "object") return false;
  const poi = value as Partial<POI>;
  return (
    typeof poi.id === "string" &&
    typeof poi.name === "string" &&
    typeof poi.description === "string" &&
    typeof poi.category === "string" &&
    Array.isArray(poi.tags) &&
    typeof poi.suggested_visit_minutes === "number" &&
    (poi.setting === "indoor" || poi.setting === "outdoor" || poi.setting === "mixed") &&
    Array.isArray(poi.source_urls)
  );
}

if (!Array.isArray(source.pois) || !source.pois.every(isPOI)) {
  throw new Error("The bundled Corfu POI dataset does not match the expected shape.");
}

export const pois: readonly POI[] = source.pois;
export const datasetLabel = `${source.dataset} · v${source.version}`;

export const categories = [...new Set(pois.map((poi) => poi.category))].sort();
export const settings: readonly ("all" | POI["setting"])[] = [
  "all",
  "indoor",
  "outdoor",
  "mixed",
];
