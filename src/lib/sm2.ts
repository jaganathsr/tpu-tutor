// SM-2 spaced repetition (Wozniak, 1990), with the four-button grading used by
// most flashcard apps mapped onto SM-2's 0-5 quality scale.

export type Grade = 1 | 2 | 3 | 4; // Again, Hard, Good, Easy

export interface CardState {
  ef: number;        // easiness factor, >= 1.3
  interval: number;  // days
  reps: number;      // consecutive successful reviews
  lapses: number;
  due: number;       // epoch ms
  last: number;      // epoch ms of last review
}

const DAY = 86_400_000;
const QUALITY: Record<Grade, number> = { 1: 1, 2: 3, 3: 4, 4: 5 };

export const GRADE_LABELS: Record<Grade, string> = { 1: 'Again', 2: 'Hard', 3: 'Good', 4: 'Easy' };

export function newCard(now = Date.now()): CardState {
  return { ef: 2.5, interval: 0, reps: 0, lapses: 0, due: now, last: 0 };
}

export function review(card: CardState, grade: Grade, now = Date.now()): CardState {
  const q = QUALITY[grade];
  let { ef, interval, reps, lapses } = card;

  if (q < 3) {
    reps = 0;
    interval = 0; // relearn today: shown again in the current session
    lapses += 1;
  } else {
    reps += 1;
    if (reps === 1) interval = grade === 4 ? 3 : 1;
    else if (reps === 2) interval = grade === 2 ? 3 : 6;
    else interval = Math.round(interval * ef * (grade === 2 ? 0.8 : grade === 4 ? 1.3 : 1));
  }
  ef = Math.max(1.3, ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)));

  return { ef, interval, reps, lapses, due: now + interval * DAY, last: now };
}

export function isDue(card: CardState | undefined, now = Date.now()): boolean {
  return !card || card.due <= now;
}

/** Interval the card would get for each grade, for button labels. */
export function previewIntervals(card: CardState | undefined, now = Date.now()): Record<Grade, number> {
  const c = card ?? newCard(now);
  return {
    1: review(c, 1, now).interval,
    2: review(c, 2, now).interval,
    3: review(c, 3, now).interval,
    4: review(c, 4, now).interval,
  };
}

export function fmtInterval(days: number): string {
  if (days <= 0) return 'now';
  if (days < 30) return `${days} d`;
  if (days < 365) return `${Math.round(days / 30)} mo`;
  return `${(days / 365).toFixed(1)} y`;
}
