// Loads course structure, chapter MDX, quizzes, flashcards and exams from /content.
// Adding a chapter = adding content/chapters/<id>/{index.mdx,quiz.yaml,cards.yaml}
// and listing <id> in content/course.yaml.
import type { ComponentType } from 'preact';
import course from '../../content/course.yaml';
import exams from '../../content/exams.yaml';

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
}

type MdxModule = { default: ComponentType<{ components?: Record<string, unknown> }>; frontmatter: Record<string, any> };

const mdx = import.meta.glob<MdxModule>('../../content/chapters/*/index.mdx', { eager: true });
const quizFiles = import.meta.glob<Question[]>('../../content/chapters/*/quiz.yaml', { eager: true, import: 'default' });
const cardFiles = import.meta.glob<Card[]>('../../content/chapters/*/cards.yaml', { eager: true, import: 'default' });

const dirOf = (path: string) => path.split('/').slice(-2, -1)[0];

const byDir = <T,>(files: Record<string, T>) =>
  Object.fromEntries(Object.entries(files).map(([p, v]) => [dirOf(p), v]));

const mdxByDir = byDir(mdx);
const quizByDir = byDir(quizFiles);
const cardsByDir = byDir(cardFiles);

export const chapters: ChapterMeta[] = (course.chapters as any[]).map((c) => {
  const fm = mdxByDir[c.id]?.frontmatter ?? {};
  return {
    objectives: [],
    teachback: [],
    sources: [],
    ...c,
    ...fm,
    written: Boolean(mdxByDir[c.id]),
  };
});

export const courseInfo = course as { title: string; summary: string; days: { day: number; theme: string }[] };

export function chapter(id: string) {
  return chapters.find((c) => c.id === id);
}

export function chapterContent(id: string) {
  return mdxByDir[id]?.default;
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

/** Draw a fresh question set for an exam attempt. */
export function drawExam(def: ExamDef): Question[] {
  const pool = def.chapters.flatMap(questions);
  const picked: Question[] = [];
  if (def.per_chapter_min) {
    for (const ch of def.chapters) picked.push(...shuffle(questions(ch)).slice(0, def.per_chapter_min));
  }
  for (const q of shuffle(pool)) {
    if (picked.length >= def.count) break;
    if (!picked.includes(q)) picked.push(q);
  }
  if (def.earlier) {
    const first = Math.min(...def.chapters.map((id) => chapter(id)?.number ?? 99));
    const earlier = chapters.filter((c) => c.number < first).flatMap((c) => questions(c.id));
    picked.push(...shuffle(earlier).slice(0, def.earlier));
  }
  return picked;
}
