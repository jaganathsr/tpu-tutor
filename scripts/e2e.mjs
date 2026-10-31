// End-to-end check of the quiz, flashcard and teach-back flows, keyboard-driven.
// Usage: npm run build && npm run preview & npm run test:e2e
import { chromium } from 'playwright-core';
import { readFileSync } from 'node:fs';
import { load } from 'js-yaml';
// Needs a running preview (npm run build && npm run preview) and a local Chrome.
const b = await chromium.launch({ executablePath: process.env.CHROME ?? '/usr/bin/google-chrome' });
const p = await b.newPage();
const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => m.type()==='error' && errs.push(m.text()));
await p.goto('http://localhost:4173/ch/ch01/quiz/');
// Answer every question from the YAML: correct option, exact number, or a sentence.
const quiz = load(readFileSync('content/chapters/ch01/quiz.yaml', 'utf8'));
const verdicts = [];
for (const q of quiz) {
  if (q.options) {
    await p.keyboard.press(String(q.options.findIndex((o) => o.correct) + 1));
    await p.keyboard.press('Enter');
    verdicts.push(await p.locator('.option.is-correct .verdict').first().textContent());
  } else if (q.type === 'numeric') {
    await p.fill('.numeric-row input', String(q.answer));
    await p.keyboard.press('Enter');
    verdicts.push(await p.locator('.result .verdict').textContent());
  } else {
    await p.fill('textarea', 'An answer.');
    await p.keyboard.press('Control+Enter');
    await p.keyboard.press('2');
    verdicts.push('self-graded partly');
  }
  await p.keyboard.press('Enter');
}
const score = await p.locator('.quiz p.ui').textContent();
const [v1, v2, v3, v4] = verdicts;
await p.screenshot({ path: 'docs/screenshots/quiz-result-light.png' });
// flashcards
await p.goto('http://localhost:4173/ch/ch01/cards/');
await p.keyboard.press(' ');
const ivl = await p.locator('.grades .ivl').allTextContents();
await p.screenshot({ path: 'docs/screenshots/cards-flipped-light.png' });
await p.keyboard.press('3'); await p.waitForTimeout(200);
const left = await p.locator('.deck-head').textContent();
// teach-back
await p.goto('http://localhost:4173/ch/ch01/teach/');
await p.check('.objectives input >> nth=0'); await p.fill('textarea', 'Roofline: a decode step at batch 64 is HBM-bound on Ironwood.');
await p.click('text=Save attempt');
const att = await p.locator('.attempt').count();
await p.goto('http://localhost:4173/');
const summary = await p.locator('.summary-line').textContent();
const state = await p.evaluate(() => localStorage.getItem('tpudd:v1'));
console.log({ v1, v2, v3, v4, score, ivl, left, att, summary, stateLen: state.length, errs });
await b.close();
