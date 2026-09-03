// Mirrors the schema documented in SPEC.md 3 (corpus/catalog.json). Kept as a hand-written
// type since the catalog itself is hand-curated data, not generated.
export interface CatalogEntry {
  id: string;
  title: string;
  type: string;
  published: string;
  url: string | null;
  media: { kind: "pdf"; pages: number } | { kind: "video"; minutes: number } | { kind: "audio"; minutes: number };

  topics: string[];
  vehicle_classes: number[];
  duty_cycles: string[];
  fleets_studied: string[];

  supersedes: string[] | null;
  superseded_by: string | null;
  status: string;

  abstract: string;
  key_findings: string[];
  data_available: string[];

  token_count: number;
  ingested: string;
  ingest_model: string;
}
