// Builds src/generated/sources.json from the tables in research/sources.md,
// so research/sources.md stays the single list of citable sources.
// Run automatically by `npm run dev` and `npm run build`.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const text = readFileSync('research/sources.md', 'utf8');
const sources = {};
// Hand-set short titles where the automatic shortening reads badly.
const SHORT = {
  'G-7x': 'Ironwood: the first Google TPU for the age of inference',
  'T-how': 'TorchTPU docs: How TorchTPU works',
  'T-exec': 'TorchTPU docs: Execution model',
  'T-dist': 'TorchTPU docs: Distributed support',
  'T-kern': 'TorchTPU docs: Custom kernels',
};
let section = '';

for (const line of text.split('\n')) {
  const h = line.match(/^#{2,3}\s+(.*)/);
  if (h) section = h[1].replace(/^\d+(\.\d+)?\s*/, '');
  if (!line.startsWith('|')) continue;
  const cells = line.split('|').slice(1, -1).map((c) => c.trim());
  const id = cells[0];
  if (!/^[A-Z][A-Za-z0-9]*(-[A-Za-z0-9]+)*$/.test(id) || id === 'ID') continue;
  const url = (line.match(/https?:\/\/[^\s)|`]+/) || [])[0];
  if (!url) continue;
  const full = cells[1].replace(/\*\*/g, '').replace(/https?:\/\/\S+/g, '').trim();
  // Short title for sidebars. Papers and posts: authors plus the quoted title.
  // Everything else: the text before any notes, with long parentheticals removed.
  let title;
  const quoted = full.match(/^[^"(]*"[^"]+"/);
  if (quoted) title = quoted[0];
  else {
    let t = full;
    while (/\([^()]*\)/.test(t)) t = t.replace(/\s*\([^()]*\)/g, '');
    title = t.split(/: |; |\.["”]? (?=[A-Z])/)[0];
    if (!title.includes(' ')) title = full.split(/: /)[0];
  }
  title = title.replace(/[.,]+"$/, '"').replace(/[.,]+$/, '').trim();
  if (SHORT[id]) title = SHORT[id];
  sources[id] = { id, title, full, url: url.replace(/[.,]$/, ''), section };
}

mkdirSync('src/generated', { recursive: true });
writeFileSync('src/generated/sources.json', JSON.stringify(sources, null, 1));
console.log(`sources.json: ${Object.keys(sources).length} sources`);
