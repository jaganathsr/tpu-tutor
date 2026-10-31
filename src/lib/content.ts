// Loads course structure, chapter MDX, quizzes, flashcards and exams from /content.
// Adding a chapter = adding content/chapters/<id>/{index.mdx,quiz.yaml,cards.yaml}
// and listing <id> in content/course.yaml.
import type { ComponentType } from 'preact';
import course from '../../content/course.yaml';
import exams from '../../content/exams.yaml';
import chaptersMeta from '../generated/chapters-meta.json';

export interface SourceRef { id: string; anchor?: string; label?: string }

export interface ChapterMeta {
  id: string;
  number: number;
  day: number;
  block: number;
  title: string;
  minutes: number;
  written: boolean;
  status?: string;
  description?: string;
  objectives: string[];
  teachback: string[];
  sources: SourceRef[];
}

export interface Option { text: string; correct?: boolean; why: string }

export interface Question {
  id: string;
  chapter: string;
  type: 'mcq' | 'numeric' | 'scenario' | 'free';
  prompt: string;
  options?: Option[];
  answer?: number;
  tolerance?: number;      // relative, default 0.05
  unit?: string;
  explanation?: string;    // HTML; shown after answering
  model_answer?: string;   // free response
  source?: string;         // e.g. "SB1 #matrix-multiplication"
  adapted_from?: string;   // e.g. "SB1 Q2"
}

export interface Card { id: string; chapter: string; front: string; back: string; source?: string }

export interface ExamDef {
  id: string;
  title: string;
  description: string;
  chapters: string[];       // primary pool
  count: number;
  earlier?: number;         // extra questions drawn from chapters before the primary pool
  per_chapter_min?: number;
  mix?: Partial<Record<Question['type'], number>>; // minimum count per question type
}

type MdxModule = { default: ComponentType<{ components?: Record<string, unknown> }>; frontmatter: Record<string, any> };

// Chapter and slide MDX are code-split: one chunk each, loaded on demand.
// Frontmatter comes from src/generated/chapters-meta.json (scripts/build-sources.mjs).
const mdxLoaders = import.meta.glob<MdxModule>('../../content/chapters/*/index.mdx');
const slideLoaders = import.meta.glob<MdxModule>('../../content/chapters/*/slides.mdx');
const quizFiles = import.meta.glob<Question[]>('../../content/chapters/*/quiz.yaml', { eager: true, import: 'default' });
const cardFiles = import.meta.glob<Card[]>('../../content/chapters/*/cards.yaml', { eager: true, import: 'default' });

const dirOf = (path: string) => path.split('/').slice(-2, -1)[0];

const byDir = <T,>(files: Record<string, T>) =>
  Object.fromEntries(Object.entries(files).map(([p, v]) => [dirOf(p), v]));

const quizByDir = byDir(quizFiles);
const cardsByDir = byDir(cardFiles);
const loadersByDir = byDir(mdxLoaders);
const slideLoadersByDir = byDir(slideLoaders);
const loadedMdx: Record<string, MdxModule> = {};
const loadedSlides: Record<string, MdxModule> = {};

export const chapters: ChapterMeta[] = (course.chapters as any[]).map((c) => {
  const fm = (chaptersMeta as Record<string, any>)[c.id] ?? {};
  return {
    objectives: [],
    teachback: [],
    sources: [],
    ...c,
    ...fm,
    written: Boolean(chaptersMeta[c.id as keyof typeof chaptersMeta]),
  };
});

// Every deploy renames the lazily loaded chunks, so a page opened before a deploy
// asks for files that no longer exist. Reload once to pick up the new build; if
// that already happened in the last minute, let the caller show an error.
const RELOAD_KEY = 'tpudd:chunk-reload';
function importOrReload<T>(load: () => Promise<T>): Promise<T> {
  return load().catch((err) => {
    if (typeof location === 'undefined') throw err;
    let last = 0;
    try { last = Number(sessionStorage.getItem(RELOAD_KEY)) || 0; } catch { /* storage blocked */ }
    if (Date.now() - last < 60_000) throw err;
    try { sessionStorage.setItem(RELOAD_KEY, String(Date.now())); } catch { /* storage blocked */ }
    location.reload();
    return new Promise<T>(() => {});
  });
}

export async function loadChapter(id: string) {
  if (!loadedMdx[id] && loadersByDir[id]) loadedMdx[id] = await importOrReload(loadersByDir[id]);
  return loadedMdx[id]?.default;
}

export async function loadSlides(ids: string[]) {
  await Promise.all(ids.map(async (id) => {
    if (!loadedSlides[id] && slideLoadersByDir[id]) loadedSlides[id] = await importOrReload(slideLoadersByDir[id]);
  }));
}

/** Load whatever the page at `path` renders, before first render (client boot and prerender). */
export async function preloadForPath(path: string) {
  const parts = path.split('/').filter(Boolean);
  if (parts[0] === 'ch' && parts[1]) await loadChapter(parts[1]);
  if (parts[0] === 'slides') {
    const scope = parts[1] ?? 'all';
    await loadSlides(scope === 'all' ? chapters.filter((c) => c.written).map((c) => c.id) : [scope]);
  }
}

const pageFiles = import.meta.glob<MdxModule>('../../content/pages/*.mdx', { eager: true });

/** Site pages (About, Privacy, Terms) from content/pages/<slug>.mdx. */
export const sitePages = Object.fromEntries(
  Object.entries(pageFiles).map(([p, m]) => [p.split('/').pop()!.replace(/\.mdx$/, ''), { Content: m.default, ...(m.frontmatter as { title: string; description: string; updated: string }) }]),
);

export const courseInfo = course as { title: string; summary: string; days: { day: number; theme: string }[] };

export function chapter(id: string) {
  return chapters.find((c) => c.id === id);
}

/** The chapter's MDX component if already loaded (see loadChapter / preloadForPath). */
export function chapterContent(id: string) {
  return loadedMdx[id]?.default;
}

/** Whether the chapter has slides, from build-time metadata. */
export function hasSlides(id: string) {
  return Boolean((chaptersMeta as Record<string, any>)[id]?.hasSlides);
}

/** The chapter's slides component if already loaded. */
export function chapterSlides(id: string) {
  return loadedSlides[id]?.default;
}

export function questions(id: string): Question[] {
  return (quizByDir[id] ?? []).map((q) => ({ ...q, chapter: id }));
}

export function cards(id: string): Card[] {
  return (cardsByDir[id] ?? []).map((c) => ({ ...c, chapter: id }));
}

export function allCards(): Card[] {
  return chapters.flatMap((c) => cards(c.id));
}

export const examDefs: ExamDef[] = exams;

function shuffle<T>(xs: T[]): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Draw a fresh question set for an exam attempt: per-chapter minimums first,
 *  then per-type minimums, then fill at random. */
export function drawExam(def: ExamDef): Question[] {
  const pool = shuffle(def.chapters.flatMap(questions));
  const picked: Question[] = [];
  const take = (q: Question) => { if (!picked.includes(q) && picked.length < def.count) picked.push(q); };
  if (def.per_chapter_min) {
    for (const ch of def.chapters) shuffle(questions(ch)).slice(0, def.per_chapter_min).forEach(take);
  }
  for (const [type, min] of Object.entries(def.mix ?? {}) as [Question['type'], number][]) {
    const have = picked.filter((q) => q.type === type).length;
    pool.filter((q) => q.type === type && !picked.includes(q)).slice(0, Math.max(0, min - have)).forEach((q) => {
      if (picked.length < def.count) picked.push(q);
      else {
        // Swap out a question of an over-represented type to make room.
        const i = picked.findIndex((x) => x.type === 'mcq');
        if (i >= 0) picked[i] = q;
      }
    });
  }
  for (const q of pool) take(q);
  if (def.earlier) {
    const first = Math.min(...def.chapters.map((id) => chapter(id)?.number ?? 99));
    const earlier = chapters.filter((c) => c.number < first).flatMap((c) => questions(c.id));
    picked.push(...shuffle(earlier).slice(0, def.earlier));
  }
  return shuffle(picked);
}
