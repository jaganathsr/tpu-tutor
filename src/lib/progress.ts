// Derived progress figures and the reading-time tracker.
import { useEffect } from 'preact/hooks';
import { allCards, chapters, questions } from './content';
import { chapterState, mutChapter, update, type State } from './storage';
import { isDue } from './sm2';

export const PASS = 0.8; // quiz best score that counts as "done"

export function chapterProgress(s: State, id: string) {
  const cs = chapterState(s, id);
  const best = cs.quiz.reduce((m, a) => Math.max(m, a.score / a.total), 0);
  return {
    read: Boolean(cs.readAt),
    quizBest: cs.quiz.length ? best : null,
    quizDone: cs.quiz.length > 0 && best >= PASS,
    teachDone: cs.teach.length > 0,
    timeMs: cs.timeMs,
    questionCount: questions(id).length,
  };
}

export function dueCount(s: State, now = Date.now()) {
  return allCards().filter((c) => s.cards[c.id] && isDue(s.cards[c.id], now)).length;
}

export function newCardCount(s: State) {
  return allCards().filter((c) => !s.cards[c.id]).length;
}

export function totals(s: State) {
  const ps = chapters.map((c) => chapterProgress(s, c.id));
  const scored = ps.filter((p) => p.quizBest != null);
  return {
    read: ps.filter((p) => p.read).length,
    timeMs: ps.reduce((t, p) => t + p.timeMs, 0),
    quizAvg: scored.length ? scored.reduce((t, p) => t + (p.quizBest ?? 0), 0) / scored.length : null,
  };
}

/** Counts time while the tab is visible and the learner has interacted in the last 2 minutes. */
export function useTimeTracker(chapterId: string | undefined) {
  useEffect(() => {
    if (!chapterId) return;
    let lastInput = Date.now();
    const bump = () => (lastInput = Date.now());
    const evs = ['scroll', 'keydown', 'pointerdown', 'pointermove'];
    evs.forEach((e) => window.addEventListener(e, bump, { passive: true }));
    const TICK = 15_000;
    const t = setInterval(() => {
      if (document.visibilityState !== 'visible' || Date.now() - lastInput > 120_000) return;
      update((s) => {
        mutChapter(s, chapterId).timeMs += TICK;
      });
    }, TICK);
    return () => {
      clearInterval(t);
      evs.forEach((e) => window.removeEventListener(e, bump));
    };
  }, [chapterId]);
}
