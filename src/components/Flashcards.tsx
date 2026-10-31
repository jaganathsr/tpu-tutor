// Flashcard review session with SM-2 scheduling.
// Keys: Space flips the card; 1-4 grade it (Again, Hard, Good, Easy).
import { useLayoutEffect, useMemo, useRef, useState } from 'preact/hooks';
import type { Card } from '../lib/content';
import { chapter } from '../lib/content';
import { getState, update } from '../lib/storage';
import { GRADE_LABELS, fmtInterval, isDue, previewIntervals, review, type Grade } from '../lib/sm2';

const NEW_PER_SESSION = 25;

export function buildQueue(cards: Card[], now = Date.now()): Card[] {
  const s = getState();
  const due = cards.filter((c) => s.cards[c.id] && isDue(s.cards[c.id], now)).sort((a, b) => s.cards[a.id].due - s.cards[b.id].due);
  const fresh = cards.filter((c) => !s.cards[c.id]).slice(0, NEW_PER_SESSION);
  return [...due, ...fresh];
}

export function Flashcards({ cards, showChapter = false }: { cards: Card[]; showChapter?: boolean }) {
  const initial = useMemo(() => buildQueue(cards), [cards]);
  const [queue, setQueue] = useState<Card[]>(initial);
  const [flipped, setFlipped] = useState(false);
  const [reviewed, setReviewed] = useState(0);

  const card = queue[0];
  const state = card ? getState().cards[card.id] : undefined;
  const ivl = card ? previewIntervals(state) : null;

  function grade(g: Grade) {
    if (!card || !flipped) return;
    const next = review(state ?? { ef: 2.5, interval: 0, reps: 0, lapses: 0, due: Date.now(), last: 0 }, g);
    update((s) => {
      s.cards[card.id] = next;
    });
    setReviewed((n) => n + 1);
    // "Again" puts the card back a few places in this session.
    setQueue((q) => {
      const rest = q.slice(1);
      if (g === 1) rest.splice(Math.min(3, rest.length), 0, card);
      return rest;
    });
    setFlipped(false);
  }

  // One listener for the component's lifetime; it always calls the latest handler,
  // so fast key sequences never hit a stale render.
  const onKeyRef = useRef<(e: KeyboardEvent) => void>(() => {});
  onKeyRef.current = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.matches?.('input, textarea, select')) return;
      if (e.key === ' ') {
        e.preventDefault();
        setFlipped(true);
      } else if (flipped && /^[1-4]$/.test(e.key)) {
        grade(+e.key as Grade);
      }
    };
  useLayoutEffect(() => {
    const onKey = (e: KeyboardEvent) => onKeyRef.current(e);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  if (!cards.length) return <p class="empty">No flashcards yet.</p>;

  if (!card) {
    const next = cards.map((c) => getState().cards[c.id]?.due).filter(Boolean).sort()[0];
    return (
      <div class="empty">
        {reviewed ? `Session done: ${reviewed} reviews. ` : 'No cards are due. '}
        {next ? `The next card is due ${new Date(next).toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' })}.` : ''}
      </div>
    );
  }

  const newCount = queue.filter((c) => !getState().cards[c.id]).length;
  const ch = chapter(card.chapter);

  return (
    <div>
      <div class="deck-head">
        <span>
          <b>{queue.length}</b> left in this session, <b>{newCount}</b> of them new.
        </span>
        <span>Reviewed <b>{reviewed}</b></span>
      </div>
      <div class="flashcard" aria-live="polite">
        <div class="side-label">
          {showChapter && ch ? `Chapter ${ch.number}. ` : ''}
          {flipped ? 'Answer' : 'Question'}
        </div>
        <div class="front" dangerouslySetInnerHTML={{ __html: card.front }} />
        {flipped ? <div class="back prose" dangerouslySetInnerHTML={{ __html: card.back }} /> : null}
        {flipped && card.source ? <div class="card-src">Source: {card.source}</div> : null}
      </div>
      {!flipped ? (
        <div class="btn-row">
          <button class="btn primary" onClick={() => setFlipped(true)}>
            Show answer<kbd>Space</kbd>
          </button>
        </div>
      ) : (
        <div class="grades" role="group" aria-label="Grade your recall">
          {([1, 2, 3, 4] as Grade[]).map((g) => (
            <button class={`btn ${g === 3 ? 'primary' : ''}`} onClick={() => grade(g)}>
              <span>
                {GRADE_LABELS[g]} <kbd>{g}</kbd>
              </span>
              <span class="ivl">{fmtInterval(ivl![g])}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
