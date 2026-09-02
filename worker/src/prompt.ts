// Safe prompt-template filling.
//
// The obvious `template.replace("{{question}}", value)` is wrong in two ways that both bit
// this project (see the security review that added this file):
//
//  1. With a *string* replacement, `$&`, `` $` ``, `$'` and `$1` in the REPLACEMENT are
//     substitution patterns, not literal text. Six corpus sources contain `$'`, which expands
//     to "everything after the match" -- splicing the tail of the prompt into the middle of a
//     source document. A question containing `` $` `` expands to "everything before the
//     match", duplicating the entire ~170K-token catalog into the question slot and doubling
//     the cost of the request.
//  2. Chained `.replace()` calls fill placeholders in sequence, so a value substituted early
//     is itself scanned for later placeholders -- a document containing the literal
//     "{{question}}" would swallow the real question slot.
//
// One regex pass with a *function* replacer fixes both: function replacements never expand
// `$` patterns, and each placeholder is visited exactly once against the original template.
export function fillTemplate(template: string, values: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (match, key: string) =>
    Object.prototype.hasOwnProperty.call(values, key) ? values[key] : match,
  );
}

/** Placeholders the template declares but the caller didn't supply -- a build/deploy-time
 * mistake worth catching loudly rather than shipping a prompt with a literal "{{question}}"
 * in it. */
export function missingPlaceholders(template: string, values: Record<string, string>): string[] {
  const found = new Set<string>();
  for (const m of template.matchAll(/\{\{(\w+)\}\}/g)) {
    if (!Object.prototype.hasOwnProperty.call(values, m[1])) found.add(m[1]);
  }
  return [...found];
}
