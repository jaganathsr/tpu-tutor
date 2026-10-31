// Checks WCAG contrast of the design tokens in src/styles/tokens.css,
// for the light set (:root) and the dark set (:root[data-theme='dark']).
import { readFileSync } from 'node:fs';

const css = readFileSync('src/styles/tokens.css', 'utf8');

function block(selector) {
  const i = css.indexOf(selector);
  const body = css.slice(css.indexOf('{', i) + 1, css.indexOf('}', i));
  return Object.fromEntries([...body.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})/gi)].map((m) => [m[1], m[2]]));
}

const lum = (hex) => {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
};
const ratio = (a, b) => {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};

// [foreground, background, minimum]
const pairs = [
  ['ink', 'paper', 7], ['ink-2', 'paper', 7], ['ink', 'paper-2', 7],
  ['ink', 'callout', 7], ['ink', 'breaks', 7], ['ink', 'accent-tint', 7], ['ink-3', 'paper', 3],
  ['accent', 'paper', 4.5], ['on-accent', 'accent-strong', 4.5], ['accent', 'accent-tint', 4.5],
  ['ok', 'ok-tint', 4.5], ['bad', 'bad-tint', 4.5], ['flag', 'flag-tint', 4.5],
  ['ok', 'paper', 4.5], ['bad', 'paper', 4.5],
  ['s1', 'paper-2', 3], ['s2', 'paper-2', 3], ['s3', 'paper-2', 3],
  ['day1', 'paper', 3], ['day2', 'paper', 3], ['day3', 'paper', 3], ['day4', 'paper', 3], ['day5', 'paper', 3], ['day6', 'paper', 3],
];

let failed = 0;
for (const [name, sel] of [['light', ':root {'], ['dark', ":root[data-theme='dark']"]]) {
  const t = block(sel);
  for (const [fg, bg, min] of pairs) {
    const r = ratio(t[fg], t[bg]);
    const ok = r >= min;
    if (!ok) failed++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(5)} ${fg.padEnd(7)} on ${bg.padEnd(11)} ${r.toFixed(2)} (min ${min})`);
  }
}
process.exit(failed ? 1 : 0);
