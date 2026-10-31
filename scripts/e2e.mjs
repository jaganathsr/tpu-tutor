// End-to-end check of the quiz, flashcard and teach-back flows, keyboard-driven.
// Usage: npm run build && npm run preview & npm run test:e2e
import { chromium } from 'playwright-core';
// Needs a running preview (npm run build && npm run preview) and a local Chrome.
const b = await chromium.launch({ executablePath: process.env.CHROME ?? '/usr/bin/google-chrome' });
const p = await b.newPage();
const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => m.type()==='error' && errs.push(m.text()));
await p.goto('http://localhost:4173/#/ch/ch01/quiz');
await p.keyboard.press('1'); await p.keyboard.press('Enter');
const v1 = await p.locator('.option.is-correct .verdict').first().textContent();
await p.keyboard.press('Enter');
await p.fill('.numeric-row input', '313'); await p.keyboard.press('Enter');
const v2 = await p.locator('.result .verdict').textContent();
await p.keyboard.press('Enter');
await p.fill('.numeric-row input', '1.2e2'); await p.keyboard.press('Enter');
const v3 = await p.locator('.result .verdict').textContent();
await p.keyboard.press('Enter');
await p.keyboard.press('2'); await p.keyboard.press('Enter');
const v4 = await p.locator('.option.is-wrong .verdict').textContent();
await p.keyboard.press('Enter');
await p.fill('textarea', 'int8 weights halve bytes'); await p.keyboard.press('Control+Enter');
await p.keyboard.press('1'); await p.keyboard.press('Enter');
const score = await p.locator('.quiz p.ui').textContent();
await p.screenshot({ path: 'docs/screenshots/quiz-result-light.png' });
// flashcards
await p.goto('http://localhost:4173/#/ch/ch01/cards');
await p.keyboard.press(' ');
const ivl = await p.locator('.grades .ivl').allTextContents();
await p.screenshot({ path: 'docs/screenshots/cards-flipped-light.png' });
await p.keyboard.press('3'); await p.waitForTimeout(200);
const left = await p.locator('.deck-head').textContent();
// teach-back
await p.goto('http://localhost:4173/#/ch/ch01/teach');
await p.check('.objectives input >> nth=0'); await p.fill('textarea', 'Roofline: a decode step at batch 64 is HBM-bound on Ironwood.');
await p.click('text=Save attempt');
const att = await p.locator('.attempt').count();
await p.goto('http://localhost:4173/#/');
const summary = await p.locator('.summary-line').textContent();
const state = await p.evaluate(() => localStorage.getItem('tpudd:v1'));
console.log({ v1, v2, v3, v4, score, ivl, left, att, summary, stateLen: state.length, errs });
await b.close();
