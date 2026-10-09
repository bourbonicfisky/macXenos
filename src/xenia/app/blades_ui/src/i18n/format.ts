/**
 * Message formatting: "{name}" placeholders, and ICU-style plurals for the
 * languages whose plural rules differ from English's:
 *
 *   "{count, plural, one {# game} other {# games}}"
 *
 * "#" in a branch is the count, formatted for the locale. Branches are
 * Intl.PluralRules categories (zero, one, two, few, many, other) or exact
 * values (=0). Pure, so the tests can run it in Node.
 */

export type MessageParams = Record<string, string | number>;

/** The placeholder names a message uses, for checking translations keep them. */
export function placeholders(message: string): string[] {
  const names = new Set<string>();
  let depth = 0;
  for (let i = 0; i < message.length; i++) {
    if (message[i] === "{") {
      if (depth === 0) {
        const match = /^\{\s*([A-Za-z_][A-Za-z0-9_]*)/.exec(message.slice(i));
        if (match) names.add(match[1]);
      }
      depth++;
    } else if (message[i] === "}") {
      depth--;
    }
  }
  return [...names].sort();
}

/** The text between the brace at `open` and its partner. */
function enclosed(text: string, open: number): { inner: string; end: number } {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    if (text[i] === "{") depth++;
    else if (text[i] === "}" && --depth === 0) return { inner: text.slice(open + 1, i), end: i };
  }
  throw new Error(`unbalanced braces in "${text}"`);
}

function plural(body: string, value: number, locale: string, params: MessageParams): string {
  const branches = new Map<string, string>();
  const pattern = /\s*(=\d+|zero|one|two|few|many|other)\s*\{/g;
  let at = 0;
  while (at < body.length) {
    pattern.lastIndex = at;
    const match = pattern.exec(body);
    if (!match || match.index !== at) break;
    const open = match.index + match[0].length - 1;
    const { inner, end } = enclosed(body, open);
    branches.set(match[1], inner);
    at = end + 1;
    while (at < body.length && /\s/.test(body[at])) at++;
  }
  const category = new Intl.PluralRules(locale).select(value);
  const chosen = branches.get(`=${value}`) ?? branches.get(category) ?? branches.get("other") ?? "";
  const counted = chosen.replace(/#/g, new Intl.NumberFormat(locale).format(value));
  return format(counted, params, locale);
}

export function format(message: string, params: MessageParams = {}, locale = "en"): string {
  let out = "";
  for (let i = 0; i < message.length; i++) {
    if (message[i] !== "{") {
      out += message[i];
      continue;
    }
    const { inner, end } = enclosed(message, i);
    const [name, kind, ...rest] = inner.split(",");
    const key = name.trim();
    if (kind?.trim() === "plural") {
      out += plural(rest.join(","), Number(params[key] ?? 0), locale, params);
    } else {
      const value = params[key];
      out += value === undefined ? `{${key}}` : typeof value === "number" ? new Intl.NumberFormat(locale).format(value) : value;
    }
    i = end;
  }
  return out;
}

const ACCENTS: Record<string, string> = {
  a: "å", b: "ƀ", c: "ç", d: "ð", e: "é", f: "ƒ", g: "ĝ", h: "ĥ", i: "î", j: "ĵ", k: "ķ", l: "ļ", m: "ɱ",
  n: "ñ", o: "ö", p: "þ", r: "ŕ", s: "š", t: "ţ", u: "û", w: "ŵ", y: "ý", z: "ž",
  A: "Å", B: "Ɓ", C: "Ç", D: "Ð", E: "É", G: "Ĝ", H: "Ĥ", I: "Î", J: "Ĵ", K: "Ķ", L: "Ļ", N: "Ñ",
  O: "Ö", R: "Ŕ", S: "Š", T: "Ţ", U: "Û", W: "Ŵ", Y: "Ý", Z: "Ž"
};

/** Accents the words of a message, leaving its placeholders and plural
    syntax as they are. */
function accent(message: string): string {
  let out = "";
  for (let i = 0; i < message.length; i++) {
    if (message[i] !== "{") {
      out += message[i] === "#" ? "#" : (ACCENTS[message[i]] ?? message[i]);
      continue;
    }
    const { inner, end } = enclosed(message, i);
    const [name, kind, ...rest] = inner.split(",");
    if (kind?.trim() !== "plural") {
      out += `{${inner}}`;
    } else {
      // Keep "name, plural," and each selector; accent each branch's text.
      let body = rest.join(",");
      let branches = "";
      const pattern = /(\s*(?:=\d+|zero|one|two|few|many|other)\s*)\{/y;
      let at = 0;
      for (;;) {
        pattern.lastIndex = at;
        const match = pattern.exec(body);
        if (!match) break;
        const open = at + match[0].length - 1;
        const branch = enclosed(body, open);
        branches += `${match[1]}{${accent(branch.inner)}}`;
        at = branch.end + 1;
      }
      body = branches + body.slice(at);
      out += `{${name},${kind},${body}}`;
    }
    i = end;
  }
  return out;
}

/**
 * The pseudo-locale (development and tests only): every letter accented and
 * the message bracketed, so text that bypasses the catalog stands out and
 * layouts meet longer text.
 */
export function pseudo(message: string): string {
  return `[${accent(message)}]`;
}
