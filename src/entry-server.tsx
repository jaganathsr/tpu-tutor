// Server entry for build-time prerendering (scripts/prerender.mjs). Renders one
// route to HTML and describes every route with the metadata its <head> needs.
import { renderToString } from 'preact-render-to-string';
import { App } from './app';
import { setSsrPath } from './lib/router';
import { chapters, courseInfo, examDefs, preloadForPath } from './lib/content';

export const preload = preloadForPath;

export function render(path: string): string {
  setSsrPath(path);
  try {
    return renderToString(<App />);
  } finally {
    setSsrPath(null);
  }
}

export interface RouteMeta {
  path: string;
  title: string;
  description: string;
  index: boolean;            // false: noindex, kept out of the sitemap
  kind: 'home' | 'chapter' | 'page';
  chapter?: { id: string; number: number; title: string; objectives: string[] };
}

export function routes(): RouteMeta[] {
  const site = 'TPU Deep Dive';
  const out: RouteMeta[] = [
    {
      path: '/',
      title: 'TPU Deep Dive: TPU hardware and software for GPU engineers',
      description:
        'A free course on Google TPUs for GPU engineers: Ironwood and TPU 8 hardware, ICI and OCS, sharding, inference, XLA, Pallas, TorchTPU and XProf.',
      index: true,
      kind: 'home',
    },
    {
      path: '/sources',
      title: `Sources | ${site}`,
      description: 'Every paper, documentation page and post cited in the TPU Deep Dive course, with verification status.',
      index: true,
      kind: 'page',
    },
  ];
  for (const c of chapters.filter((x) => x.written)) {
    out.push({
      path: `/ch/${c.id}`,
      title: `${c.title} | ${site}`,
      description: c.description ?? courseInfo.summary,
      index: true,
      kind: 'chapter',
      chapter: { id: c.id, number: c.number, title: c.title, objectives: c.objectives },
    });
    for (const tab of ['quiz', 'cards', 'teach']) {
      out.push({ path: `/ch/${c.id}/${tab}`, title: `${c.title}: ${tab} | ${site}`, description: c.description ?? '', index: false, kind: 'page' });
    }
  }
  for (const p of ['/review', '/exams', '/data', '/slides/all/0']) {
    out.push({ path: p, title: `${p.split('/')[1][0].toUpperCase()}${p.split('/')[1].slice(1)} | ${site}`, description: courseInfo.summary, index: false, kind: 'page' });
  }
  for (const e of examDefs) {
    out.push({ path: `/exam/${e.id}`, title: `${e.title} | ${site}`, description: e.description, index: false, kind: 'page' });
  }
  return out;
}

export const course = { title: courseInfo.title, summary: courseInfo.summary };
