// Review deck: every written chapter's slides in course order, or one chapter.
// Keys: Right/Left, Space, PageUp/PageDown, Home/End, F for full screen.
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import { chapters, chapterSlides, courseInfo, hasSlides, preloadForPath, type ChapterMeta } from '../lib/content';
import { href, navigate } from '../lib/router';
import { slideComponents } from '../components/slides';

function ChapterTitleSlide({ ch }: { ch: ChapterMeta }) {
  return (
    <section class="slide section-slide" data-chapter={ch.id}>
      <div class="slide-inner">
        <div class="slide-kicker">
          Chapter {ch.number}. {ch.day > 5 ? 'Synthesis' : `Day ${ch.day}, ${ch.block === 1 ? 'first' : 'second'} block`}
        </div>
        <h2 class="slide-title big">{ch.title}</h2>
        <div class="slide-body">
          <p class="slide-sub">By the end you can:</p>
          <ul>
            {ch.objectives.map((o) => (
              <li>{o.replace(/^I can /, '')}</li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

function DeckTitle({ list }: { list: ChapterMeta[] }) {
  return (
    <section class="slide section-slide deck-title" data-chapter="">
      <div class="slide-inner">
        <div class="slide-kicker">Review deck</div>
        <h2 class="slide-title big">{courseInfo.title}</h2>
        <div class="slide-body">
          <ol class="deck-toc">
            {list.map((c) => (
              <li style={{ '--dayc': `var(--day${c.day})` }}>{c.title}</li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}

export function SlidesPage({ scope, n }: { scope: string; n: number }) {
  const [, setLoaded] = useState(0);
  const wanted = chapters.filter((c) => c.written && hasSlides(c.id) && (scope === 'all' || c.id === scope));
  const list = wanted.filter((c) => chapterSlides(c.id));
  useEffect(() => {
    if (list.length < wanted.length) preloadForPath(`/slides/${scope}`).then(() => setLoaded((n) => n + 1));
  }, [scope]);
  const deckRef = useRef<HTMLDivElement>(null);
  const [count, setCount] = useState(0);
  const [label, setLabel] = useState('');
  const idx = Math.max(0, Math.min(n, Math.max(0, count - 1)));

  const go = (i: number) => navigate(`/slides/${scope}/${Math.max(0, Math.min(i, count - 1))}`);

  // Show the current slide and colour it by its chapter's day. Re-run when the
  // lazily loaded slide modules arrive (list.length), since the deck is not in
  // the DOM while they load.
  useLayoutEffect(() => {
    const slides = [...(deckRef.current?.querySelectorAll<HTMLElement>('.slide') ?? [])];
    setCount(slides.length);
    let chapterId = '';
    slides.forEach((s, i) => {
      if (s.dataset.chapter !== undefined) chapterId = s.dataset.chapter;
      const ch = chapters.find((c) => c.id === chapterId);
      s.style.setProperty('--dayc', ch ? `var(--day${ch.day})` : 'var(--accent-strong)');
      s.dataset.label = ch ? `Chapter ${ch.number}. ${ch.title}` : courseInfo.title;
      s.dataset.n = `${i + 1} / ${slides.length}`;
      s.toggleAttribute('data-active', i === idx);
      if (i === idx) setLabel(s.dataset.label);
    });
  }, [scope, idx, count, list.length]);

  const keyRef = useRef<(e: KeyboardEvent) => void>(() => {});
  keyRef.current = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement)?.matches?.('input, textarea, select')) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(idx + 1); }
    else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(idx - 1); }
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(count - 1);
    else if (e.key === 'f' || e.key === 'F') {
      if (document.fullscreenElement) document.exitFullscreen();
      else deckRef.current?.parentElement?.requestFullscreen?.();
    }
  };
  useLayoutEffect(() => {
    const on = (e: KeyboardEvent) => keyRef.current(e);
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, []);

  // Jump to a chapter's first slide.
  function jump(id: string) {
    const slides = [...(deckRef.current?.querySelectorAll<HTMLElement>('.slide') ?? [])];
    const i = slides.findIndex((s) => s.dataset.chapter === id);
    if (i >= 0) go(i);
  }

  useEffect(() => {
    document.body.classList.add('deck-mode');
    return () => document.body.classList.remove('deck-mode');
  }, []);

  if (list.length < wanted.length) return <div class="page narrow"><p class="ui muted">Loading slides.</p></div>;
  if (!list.length) {
    return (
      <div class="page narrow">
        <p class="empty">No slides yet for {scope === 'all' ? 'any chapter' : scope}. Slides are written with each chapter.</p>
      </div>
    );
  }

  return (
    <div class="deck-page">
      <div class="deck-stage">
        <div class="deck" ref={deckRef}>
          {scope === 'all' ? <DeckTitle list={list} /> : null}
          {list.map((c) => {
            const S = chapterSlides(c.id)!;
            return (
              <>
                <ChapterTitleSlide ch={c} />
                <S components={slideComponents} />
              </>
            );
          })}
        </div>
      </div>
      <div class="deck-bar">
        <button class="btn" onClick={() => go(idx - 1)} disabled={idx === 0}>Previous</button>
        <button class="btn primary" onClick={() => go(idx + 1)} disabled={idx >= count - 1}>Next</button>
        <span class="deck-pos num">
          {idx + 1} / {count}
        </span>
        <span class="deck-label">{label}</span>
        {scope === 'all' ? (
          <select aria-label="Jump to chapter" onChange={(e) => jump((e.target as HTMLSelectElement).value)} value="">
            <option value="">Jump to chapter</option>
            {list.map((c) => (
              <option value={c.id}>
                {c.number}. {c.title}
              </option>
            ))}
          </select>
        ) : (
          <a href={href('/slides/all/0')}>Whole course</a>
        )}
        <button class="btn" onClick={() => window.print()}>Print or save as PDF</button>
        <span class="keys-hint">Arrow keys to move, F for full screen.</span>
      </div>
    </div>
  );
}
