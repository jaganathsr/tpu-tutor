// Writes one static HTML file per route into dist/, each with the route's
// rendered content and its own <head> (title, description, canonical URL,
// Open Graph, Twitter card, JSON-LD). Also writes 404.html (app fallback for
// unlisted routes) and sitemap.xml. Runs after the client and SSR builds:
//   vite build && vite build --ssr src/entry-server.tsx --outDir dist-ssr && node scripts/prerender.mjs
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const SITE = (process.env.SITE_URL ?? 'https://learntpu.dev').replace(/\/$/, '');
// Umami analytics (cookieless). Injected only when a website ID is configured.
const UMAMI_ID = process.env.UMAMI_WEBSITE_ID?.trim();
const UMAMI_SRC = process.env.UMAMI_SRC ?? 'https://cloud.umami.is/script.js';
const umamiTag = UMAMI_ID ? `<script defer src="${UMAMI_SRC}" data-website-id="${UMAMI_ID}"></script>` : '';
const DIST = 'dist';
const { render, routes, course, preload } = await import(pathToFileURL(join(process.cwd(), 'dist-ssr/entry-server.js')).href);

const template = readFileSync(join(DIST, 'index.html'), 'utf8');
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const urlFor = (path) => `${SITE}${path === '/' ? '/' : path + '/'}`;
const all = routes();
const chaptersMeta = all.filter((r) => r.kind === 'chapter');

function jsonLd(r) {
  const courseNode = {
    '@type': 'Course',
    name: course.title,
    description: all[0].description,
    url: urlFor('/'),
    inLanguage: 'en',
    isAccessibleForFree: true,
    license: 'https://creativecommons.org/licenses/by/4.0/',
    educationalLevel: 'Advanced',
    about: ['Tensor Processing Unit', 'Machine learning accelerators', 'Large language model inference', 'Distributed training'],
  };
  if (r.kind === 'home') {
    return {
      '@context': 'https://schema.org',
      ...courseNode,
      hasPart: chaptersMeta.map((c) => ({ '@type': 'LearningResource', name: c.chapter.title, url: urlFor(c.path), position: c.chapter.number })),
    };
  }
  if (r.kind === 'chapter') {
    return {
      '@context': 'https://schema.org',
      '@graph': [
        {
          '@type': 'LearningResource',
          name: r.chapter.title,
          description: r.description,
          url: urlFor(r.path),
          inLanguage: 'en',
          learningResourceType: 'Lesson',
          educationalLevel: 'Advanced',
          position: r.chapter.number,
          teaches: r.chapter.objectives.map((o) => o.replace(/^I can /, '')),
          isPartOf: courseNode,
        },
        {
          '@type': 'BreadcrumbList',
          itemListElement: [
            { '@type': 'ListItem', position: 1, name: course.title, item: urlFor('/') },
            { '@type': 'ListItem', position: 2, name: `Chapter ${r.chapter.number}: ${r.chapter.title}`, item: urlFor(r.path) },
          ],
        },
      ],
    };
  }
  return null;
}

function head(r) {
  const url = urlFor(r.path);
  const tags = [
    `<title>${esc(r.title)}</title>`,
    `<meta name="description" content="${esc(r.description)}" />`,
    `<link rel="canonical" href="${url}" />`,
    r.index ? '' : '<meta name="robots" content="noindex, follow" />',
    `<meta property="og:site_name" content="${esc(course.title)}" />`,
    `<meta property="og:type" content="${r.kind === 'chapter' ? 'article' : 'website'}" />`,
    `<meta property="og:title" content="${esc(r.title)}" />`,
    `<meta property="og:description" content="${esc(r.description)}" />`,
    `<meta property="og:url" content="${url}" />`,
    `<meta property="og:image" content="${SITE}/og.png" />`,
    '<meta property="og:image:width" content="1200" />',
    '<meta property="og:image:height" content="630" />',
    `<meta property="og:image:alt" content="${esc(course.title)}: TPU hardware and software for GPU engineers" />`,
    '<meta name="twitter:card" content="summary_large_image" />',
    `<meta name="twitter:title" content="${esc(r.title)}" />`,
    `<meta name="twitter:description" content="${esc(r.description)}" />`,
    `<meta name="twitter:image" content="${SITE}/og.png" />`,
  ];
  if (umamiTag) tags.push(umamiTag);
  const ld = jsonLd(r);
  if (ld) tags.push(`<script type="application/ld+json">${JSON.stringify(ld).replace(/</g, '\\u003c')}</script>`);
  return tags.filter(Boolean).join('\n    ');
}

function page(r, body) {
  const out = template
    .replace(/<!--head:start-->[\s\S]*?<!--head:end-->/, head(r))
    .replace('<div id="app"></div>', `<div id="app">${body}</div>`);
  if (out === template) throw new Error('template markers not found');
  return out;
}

let n = 0;
for (const r of all) {
  const file = r.path === '/' ? join(DIST, 'index.html') : join(DIST, r.path, 'index.html');
  mkdirSync(dirname(file), { recursive: true });
  await preload(r.path);
  writeFileSync(file, page(r, render(r.path)));
  n++;
}

// 404.html: GitHub Pages serves it for any unlisted path; the app renders from location.pathname.
writeFileSync(
  join(DIST, '404.html'),
  template.replace(/<!--head:start-->[\s\S]*?<!--head:end-->/, `<title>TPU Deep Dive</title>\n    <meta name="robots" content="noindex" />${umamiTag ? '\n    ' + umamiTag : ''}`),
);

const today = new Date().toISOString().slice(0, 10);
const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${all.filter((r) => r.index).map((r) => `  <url><loc>${urlFor(r.path)}</loc><lastmod>${today}</lastmod><changefreq>monthly</changefreq><priority>${r.kind === 'home' ? '1.0' : r.kind === 'chapter' ? '0.8' : '0.4'}</priority></url>`).join('\n')}
</urlset>
`;
writeFileSync(join(DIST, 'sitemap.xml'), sitemap);
// robots.txt is only read at a domain root, so write it only when the site is served from one.
if (new URL(SITE).pathname === '/') writeFileSync(join(DIST, 'robots.txt'), `User-agent: *\nAllow: /\n\nSitemap: ${SITE}/sitemap.xml\n`);
console.log(`prerendered ${n} routes for ${SITE}; sitemap has ${all.filter((r) => r.index).length} URLs; umami ${UMAMI_ID ? 'on' : 'off'}`);
