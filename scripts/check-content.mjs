// Lints chapter content against the writing rules and checks citations.
//   node scripts/check-content.mjs
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { load } from 'js-yaml';

const BANNED = [
  'unlock', 'game-changer', 'game changer', 'dive deep', 'deep dive into', 'secret sauce',
  "it's important to note", 'it is important to note', "let's explore", 'let us explore',
  'fascinating', 'powerful', 'elegant', 'seamless', 'revolutionary', 'imagine',
  'great job', 'not quite', 'well done',
];
const EMOJI = /\p{Extended_Pictographic}/u;
const sources = JSON.parse(readFileSync('src/generated/sources.json', 'utf8'));
const course = load(readFileSync('content/course.yaml', 'utf8'));

let problems = 0;
const report = (file, msg) => {
  problems++;
  console.log(`${file}: ${msg}`);
};

// Strip math, code and JSX attribute strings so they are not linted as prose.
const prose = (s) =>
  s.replace(/\$\$[\s\S]*?\$\$/g, ' ').replace(/\$[^$\n]*\$/g, ' ').replace(/```[\s\S]*?```/g, ' ').replace(/`[^`]*`/g, ' ');

function lintText(file, text) {
  const p = prose(text);
  const lower = p.toLowerCase();
  for (const w of BANNED) if (lower.includes(w)) report(file, `banned phrase "${w}"`);
  // "!" outside JSX/markdown syntax. Allow "!=" and "<!--".
  const bangs = p.replace(/!=|<!--|!\[/g, '').match(/[A-Za-z0-9)]![\s"']/g);
  if (bangs) report(file, `exclamation mark: ${bangs.join(' ')}`);
  if (EMOJI.test(p)) report(file, 'emoji');
  for (const m of text.matchAll(/<Cite id="([^"]+)"/g)) if (!sources[m[1]]) report(file, `unknown source id ${m[1]}`);
}

for (const ch of course.chapters) {
  const dir = `content/chapters/${ch.id}`;
  if (!existsSync(`${dir}/index.mdx`)) continue;
  const mdx = readFileSync(`${dir}/index.mdx`, 'utf8');
  lintText(`${dir}/index.mdx`, mdx.replace(/^---[\s\S]*?---/, ''));
  const fm = load(mdx.match(/^---([\s\S]*?)---/)[1]);
  for (const s of fm.sources ?? []) if (!sources[s.id]) report(`${dir}/index.mdx`, `frontmatter source ${s.id} unknown`);

  if (existsSync(`${dir}/slides.mdx`)) lintText(`${dir}/slides.mdx`, readFileSync(`${dir}/slides.mdx`, 'utf8'));
  else if (fm.status !== 'draft') report(dir, 'no slides.mdx');
  const quiz = existsSync(`${dir}/quiz.yaml`) ? load(readFileSync(`${dir}/quiz.yaml`, 'utf8')) : [];
  const cards = existsSync(`${dir}/cards.yaml`) ? load(readFileSync(`${dir}/cards.yaml`, 'utf8')) : [];
  const ids = new Set();
  for (const q of quiz) {
    const f = `${dir}/quiz.yaml#${q.id}`;
    if (ids.has(q.id)) report(f, 'duplicate id');
    ids.add(q.id);
    lintText(f, JSON.stringify(q));
    if (!['mcq', 'numeric', 'scenario', 'free'].includes(q.type)) report(f, `bad type ${q.type}`);
    if ((q.type === 'mcq' || q.type === 'scenario')) {
      if (!q.options?.length) report(f, 'no options');
      else {
        if (q.options.filter((o) => o.correct).length !== 1) report(f, 'needs exactly one correct option');
        q.options.forEach((o, i) => { if (!o.why) report(f, `option ${i + 1} has no explanation`); });
      }
    }
    if (q.type === 'numeric' && typeof q.answer !== 'number') report(f, 'numeric answer missing');
    if (q.type === 'free' && !q.model_answer) report(f, 'free response needs model_answer');
    const srcId = (q.source ?? '').split(' ')[0];
    if (srcId && !sources[srcId]) report(f, `unknown source ${srcId}`);
  }
  for (const c of cards) {
    if (ids.has(c.id)) report(`${dir}/cards.yaml#${c.id}`, 'duplicate id');
    ids.add(c.id);
    lintText(`${dir}/cards.yaml#${c.id}`, JSON.stringify(c));
  }
  const planned = fm.status === 'draft' ? '' : quiz.length < 8 || quiz.length > 12 ? ` (quiz has ${quiz.length}, want 8-12)` : '';
  if (planned) report(dir, planned.trim());
  if (fm.status !== 'draft' && (cards.length < 15 || cards.length > 25)) report(dir, `cards: ${cards.length}, want 15-25`);
  if (fm.status !== 'draft' && !quiz.some((q) => q.adapted_from)) report(dir, 'no quiz question adapted from the Scaling Book');
  console.log(`${dir}: ${quiz.length} questions, ${cards.length} cards`);
}

console.log(problems ? `${problems} problems` : 'content ok');
process.exit(problems ? 1 : 0);
