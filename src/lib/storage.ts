// All learner state lives in one localStorage key. Reads and writes are wrapped
// so the app still works (without persistence) when storage is unavailable.
import { useEffect, useState } from 'preact/hooks';
import type { CardState } from './sm2';

export const KEY = 'tpudd:v1';

export interface QuizAttempt {
  at: number;
  score: number;          // points earned (free response self-graded 1 / 0.5 / 0)
  total: number;
  answers: Record<string, { correct: number; given: string }>;
}

export interface TeachAttempt {
  at: number;
  checked: string[];      // objective indices, as strings
  text: string;
}

export interface ChapterState {
  readAt?: number;
  timeMs: number;
  quiz: QuizAttempt[];
  teach: TeachAttempt[];
  teachDraft?: string;
}

export interface State {
  version: 1;
  chapters: Record<string, ChapterState>;
  cards: Record<string, CardState>;
  exams: Record<string, QuizAttempt[]>;
  prefs: { theme?: 'light' | 'dark' };
}

const empty = (): State => ({ version: 1, chapters: {}, cards: {}, exams: {}, prefs: {} });

function load(): State {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return empty();
    const s = JSON.parse(raw) as State;
    return { ...empty(), ...s };
  } catch {
    return empty();
  }
}

let state: State = load();
const listeners = new Set<() => void>();

export function getState(): State {
  return state;
}

export function update(fn: (s: State) => void) {
  const next = structuredClone(state);
  fn(next);
  state = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* storage unavailable: keep in memory only */
  }
  listeners.forEach((l) => l());
}

export function chapterState(s: State, id: string): ChapterState {
  return s.chapters[id] ?? { timeMs: 0, quiz: [], teach: [] };
}

export function mutChapter(s: State, id: string): ChapterState {
  s.chapters[id] ??= { timeMs: 0, quiz: [], teach: [] };
  return s.chapters[id];
}

export function useStore(): State {
  const [, force] = useState(0);
  useEffect(() => {
    const l = () => force((n) => n + 1);
    listeners.add(l);
    return () => listeners.delete(l);
  }, []);
  return state;
}

export function replaceState(next: State) {
  update((s) => Object.assign(s, empty(), next));
}

export function resetState() {
  update((s) => Object.assign(s, empty()));
}

// Other tabs: pick up changes.
try {
  window.addEventListener('storage', (e) => {
    if (e.key === KEY) {
      state = load();
      listeners.forEach((l) => l());
    }
  });
} catch {
  /* no window */
}
