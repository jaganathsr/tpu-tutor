import { defineConfig, type Plugin } from 'vite';
import preact from '@preact/preset-vite';
import mdx from '@mdx-js/rollup';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import remarkFrontmatter from 'remark-frontmatter';
import remarkMdxFrontmatter from 'remark-mdx-frontmatter';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import { load as yamlLoad } from 'js-yaml';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import rehypeStringify from 'rehype-stringify';

const katexOptions = { strict: false, trust: false };

// Markdown fields inside quiz/flashcard YAML are rendered to HTML at build time,
// so the browser never parses Markdown or runs KaTeX.
const MD_FIELDS = new Set(['prompt', 'text', 'why', 'explanation', 'model_answer', 'front', 'back', 'derivation']);

const md = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkMath)
  .use(remarkRehype)
  .use(rehypeKatex, katexOptions)
  .use(rehypeStringify);

function renderMd(src: string): string {
  const html = String(md.processSync(src)).trim();
  // Unwrap a single paragraph so short strings render inline.
  const m = html.match(/^<p>([\s\S]*)<\/p>$/);
  return m && !m[1].includes('<p>') ? m[1] : html;
}

function walk(node: unknown, key = ''): unknown {
  if (typeof node === 'string') return MD_FIELDS.has(key) ? renderMd(node) : node;
  if (Array.isArray(node)) return node.map((n) => walk(n, key));
  if (node && typeof node === 'object') {
    return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, walk(v, k)]));
  }
  return node;
}

function contentYaml(): Plugin {
  return {
    name: 'content-yaml',
    transform(code, id) {
      if (!/\.ya?ml$/.test(id)) return null;
      const data = walk(yamlLoad(code));
      return { code: `export default ${JSON.stringify(data)};`, map: null };
    },
  };
}

export default defineConfig({
  // Absolute base: pages live at nested paths (/ch/ch07/), so assets cannot be
  // relative. CI sets BASE_PATH (/ for learntpu.dev); local builds default to /.
  base: process.env.BASE_PATH ?? '/',
  define: { __ANALYTICS__: JSON.stringify(Boolean(process.env.UMAMI_WEBSITE_ID?.trim())) },
  plugins: [
    contentYaml(),
    {
      enforce: 'pre',
      ...mdx({
        jsxImportSource: 'preact',
        remarkPlugins: [remarkGfm, remarkMath, remarkFrontmatter, [remarkMdxFrontmatter, { name: 'frontmatter' }]],
        rehypePlugins: [rehypeSlug, [rehypeKatex, katexOptions]],
      }),
    },
    preact(),
  ],
});
