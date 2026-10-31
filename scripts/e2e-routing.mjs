// End-to-end check of path routing on a production build served under a base
// path, as on GitHub Pages:
//   BASE_PATH=/tpu-tutor/ npm run build && BASE_PATH=/tpu-tutor/ npx vite preview --port 4174 &
//   BASE=http://localhost:4174/tpu-tutor node scripts/e2e-routing.mjs
import { chromium } from 'playwright-core';

const B = process.env.BASE ?? 'http://localhost:4174/tpu-tutor';
const b = await chromium.launch({ executablePath: process.env.CHROME ?? '/usr/bin/google-chrome' });
const p = await b.newPage();
const errs = [];
p.on('pageerror', (e) => errs.push(e.message));
p.on('response', (r) => { if (r.status() >= 400 && !r.url().includes('favicon')) errs.push(`${r.status()} ${r.url()}`); });
const path = () => new URL(p.url()).pathname + new URL(p.url()).hash;

await p.goto(`${B}/ch/ch07/`);
await p.waitForTimeout(800);
console.log('chapter page:', await p.locator('h1').textContent(), '| interactives:', await p.locator('figure.interactive').count());

await p.evaluate(() => { window.__sameDoc = true; });
await p.locator('.chapter-nav a', { hasText: 'Next' }).click();
await p.waitForTimeout(800);
console.log('next link:', path(), '|', await p.locator('h1').textContent(), '| no reload:', await p.evaluate(() => window.__sameDoc === true));

await p.goBack();
await p.waitForTimeout(500);
console.log('back button:', path(), '|', await p.locator('h1').textContent());

await p.goto(`${B}/#/ch/ch03`);
await p.waitForTimeout(800);
console.log('legacy #/ link:', path(), '|', await p.locator('h1').textContent());

await p.goto(`${B}/ch/ch05/`);
await p.waitForTimeout(800);
await p.locator('.toc a').nth(2).click();
await p.waitForTimeout(300);
console.log('TOC anchor:', path(), '| scrolled:', await p.evaluate(() => scrollY > 0));

await p.goto(`${B}/slides/all/5/`);
await p.waitForTimeout(1200);
console.log('slides deck:', await p.locator('.deck-pos').textContent());

await p.goto(`${B}/ch/ch01/quiz/`);
await p.waitForTimeout(600);
await p.keyboard.press('1');
await p.keyboard.press('Enter');
console.log('quiz:', await p.locator('.option.is-correct .verdict').first().textContent());

console.log('errors:', errs);
await b.close();
process.exit(errs.length ? 1 : 0);
