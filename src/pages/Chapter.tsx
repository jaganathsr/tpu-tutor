import { useEffect, useRef, useState } from 'preact/hooks';
import { cards, chapter, chapterContent, chapters, hasSlides, loadChapter, questions, type ChapterMeta } from '../lib/content';
import { href } from '../lib/router';
import { chapterState, mutChapter, update, useStore } from '../lib/storage';
import { useTimeTracker } from '../lib/progress';
import { mdxComponents, sourceUrl } from '../components/mdx';
import sources from '../generated/sources.json';
import { Quiz } from '../components/Quiz';
import { Flashcards } from '../components/Flashcards';
import { TeachBack } from '../components/TeachBack';

const SRC = sources as Record<string, { title: string }>;
type Tab = 'read' | 'quiz' | 'cards' | 'teach';

function SourceList({ ch }: { ch: ChapterMeta }) {
  return (
    <ul class="src-list">
      {ch.sources.map((r) => (
        <li>
          <a href={sourceUrl(r.id, r.anchor)} target="_blank" rel="noreferrer">
            <span class="src-id">{r.id}</span>
            <span class="src-title">{SRC[r.id]?.title ?? 'Unknown source'}</span>
            {r.label ? <span class="src-anchor">{r.label}</span> : null}
          </a>
        </li>
      ))}
    </ul>
  );
}

function Toc({ root }: { root: HTMLElement | null }) {
  const [items, setItems] = useState<{ id: string; text: string; depth: number }[]>([]);
  useEffect(() => {
    if (!root) return;
    setItems(
      [...root.querySelectorAll('h2[id], h3[id]')].map((h) => ({
        id: h.id,
        text: h.textContent ?? '',
        depth: h.tagName === 'H2' ? 2 : 3,
      })),
    );
  }, [root]);
  if (!items.length) return null;
  return (
    <ul class="toc">
      {items.map((it) => (
        <li class={`depth-${it.depth}`}>
          <a href={`#${it.id}`}>{it.text}</a>
        </li>
      ))}
    </ul>
  );
}

function ChapterNav({ ch }: { ch: ChapterMeta }) {
  const i = chapters.findIndex((c) => c.id === ch.id);
  const prev = chapters[i - 1], next = chapters[i + 1];
  return (
    <nav class="chapter-nav" aria-label="Previous and next chapter">
      {prev ? <a href={href(`/ch/${prev.id}`)}>Previous: Chapter {prev.number}. {prev.title}</a> : <span />}
      {next ? <a href={href(`/ch/${next.id}`)}>Next: Chapter {next.number}. {next.title}</a> : <span />}
    </nav>
  );
}

export function ChapterPage({ id, tab }: { id: string; tab: Tab }) {
  const s = useStore();
  const ch = chapter(id);
  const [, setLoaded] = useState(0);
  const Content = chapterContent(id);
  // Chapter MDX is a separate chunk; load it when navigating here in-app.
  useEffect(() => {
    if (!chapterContent(id)) loadChapter(id).then(() => setLoaded((n) => n + 1));
  }, [id]);
  const mainRef = useRef<HTMLDivElement>(null);
  const [root, setRoot] = useState<HTMLElement | null>(null);
  useTimeTracker(ch?.written ? id : undefined);
  useEffect(() => setRoot(mainRef.current), [id, tab]);

  if (!ch) return <div class="page narrow"><p class="empty">There is no chapter called {id}.</p></div>;

  const cs = chapterState(s, id);
  const qs = questions(id);
  const cds = cards(id);
  const tabs: [Tab, string, number | null][] = [
    ['read', 'Read', null],
    ['quiz', 'Quiz', qs.length],
    ['cards', 'Flashcards', cds.length],
    ['teach', 'Teach-back', ch.objectives.length],
  ];

  return (
    <div class="page" style={{ '--dayc': `var(--day${ch.day})` }}>
      <div class="chapter">
        <nav class="chapter-left" aria-label="Chapter contents and sources">
          <div class="sticky">
            {tab === 'read' ? (
              <div class="left-section">
                <h2>In this chapter</h2>
                <Toc root={root} />
              </div>
            ) : null}
            <div class="left-section">
              <h2>Sources</h2>
              <SourceList ch={ch} />
            </div>
          </div>
        </nav>

        <main class="chapter-main" ref={mainRef}>
          <header class="chapter-head">
            <div class="kicker">
              {ch.day > 5
                ? `Chapter ${ch.number}. Synthesis, after Day 5, ${ch.minutes} minutes`
                : `Chapter ${ch.number}. Day ${ch.day}, ${ch.block === 1 ? 'first' : 'second'} block, ${ch.minutes} minutes`}
            </div>
            <h1>{ch.title}</h1>
            {ch.status === 'draft' ? <div class="draft-note">Draft: placeholder chapter for the design review. Content is partial.</div> : null}
          </header>
          <nav class="tabs" aria-label="Chapter sections">
            {tabs.map(([t, label, n]) => (
              <a href={href(`/ch/${id}${t === 'read' ? '' : '/' + t}`)} aria-current={tab === t ? 'page' : undefined}>
                {label}
                {n != null ? <span class="count">{n}</span> : null}
              </a>
            ))}
          </nav>

          {!ch.written ? (
            <p class="empty">This chapter is not written yet. It is planned in research/syllabus.md.</p>
          ) : tab === 'read' && !Content ? (
            <p class="ui muted">Loading the chapter.</p>
          ) : tab === 'read' && Content ? (
            <>
              <article class="prose">
                <Content components={mdxComponents} />
              </article>
              <section class="sources-inline">
                <h2>Sources</h2>
                <SourceList ch={ch} />
              </section>
              <div class="chapter-end">
                {cs.readAt ? (
                  <p>
                    Marked as read on {new Date(cs.readAt).toLocaleDateString()}.{' '}
                    <a href="javascript:void 0" onClick={() => update((st) => { delete mutChapter(st, id).readAt; })}>
                      Unmark
                    </a>
                  </p>
                ) : (
                  <div class="btn-row">
                    <button class="btn" onClick={() => update((st) => { mutChapter(st, id).readAt = Date.now(); })}>
                      Mark chapter as read
                    </button>
                  </div>
                )}
                <p>
                  Next: <a href={href(`/ch/${id}/quiz`)}>take the quiz</a>, then{' '}
                  <a href={href(`/ch/${id}/cards`)}>review the flashcards</a> and{' '}
                  <a href={href(`/ch/${id}/teach`)}>write your teach-back</a>.
                  {hasSlides(id) ? (
                    <>
                      {' '}
                      For a fast review later, <a href={href(`/slides/${id}/0`)}>open this chapter's slides</a>.
                    </>
                  ) : null}
                </p>
                <ChapterNav ch={ch} />
              </div>
            </>
          ) : tab === 'quiz' ? (
            <>
              {cs.quiz.length ? (
                <p class="ui muted">
                  {cs.quiz.length} earlier {cs.quiz.length === 1 ? 'attempt' : 'attempts'}, best{' '}
                  {Math.round(Math.max(...cs.quiz.map((a) => a.score / a.total)) * 100)}%.
                </p>
              ) : null}
              <Quiz
                key={id}
                questions={qs}
                onFinish={(a) => update((st) => { mutChapter(st, id).quiz.push(a); })}
              />
            </>
          ) : tab === 'cards' ? (
            <Flashcards key={id} cards={cds} />
          ) : (
            <TeachBack ch={ch} />
          )}
        </main>
        <div class="chapter-right" />
      </div>
    </div>
  );
}
