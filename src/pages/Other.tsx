import { useMemo, useState } from 'preact/hooks';
import { allCards, chapters, drawExam, examDefs } from '../lib/content';
import { getState, replaceState, resetState, update, useStore, type State } from '../lib/storage';
import { dueCount, newCardCount } from '../lib/progress';
import { href } from '../lib/router';
import { Flashcards } from '../components/Flashcards';
import { Quiz } from '../components/Quiz';
import sources from '../generated/sources.json';

export function Review() {
  const s = useStore();
  const cards = useMemo(() => allCards(), []);
  return (
    <div class="page narrow">
      <h1>Review</h1>
      <p class="ui muted">
        All chapters. {dueCount(s)} cards due, {newCardCount(s)} not yet seen. New cards are introduced 25 per session.
      </p>
      <Flashcards cards={cards} showChapter />
    </div>
  );
}

export function Exams() {
  const s = useStore();
  return (
    <div class="page narrow">
      <h1>Exams</h1>
      <p class="ui muted">Each attempt draws a new set of questions from the chapter quiz pools.</p>
      <ul class="plain-list">
        {examDefs.map((e) => {
          const att = s.exams[e.id] ?? [];
          return (
            <li>
              <span>
                <a href={href(`/exam/${e.id}`)}>{e.title}</a>
                <br />
                <span class="meta">{e.description}</span>
              </span>
              <span class="meta">
                {att.length ? `last ${Math.round((att[att.length - 1].score / att[att.length - 1].total) * 100)}%` : 'not taken'}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function Exam({ id }: { id: string }) {
  const def = examDefs.find((e) => e.id === id);
  const [seed, setSeed] = useState(0);
  const qs = useMemo(() => (def ? drawExam(def) : []), [def, seed]);
  if (!def) return <div class="page narrow"><p class="empty">There is no exam called {id}.</p></div>;
  const written = def.chapters.filter((c) => chapters.find((x) => x.id === c)?.written).length;
  return (
    <div class="page narrow">
      <h1>{def.title}</h1>
      <p class="ui muted">
        {def.description} {qs.length} questions drawn{written < def.chapters.length ? `; ${def.chapters.length - written} of its chapters are not written yet, so the pool is smaller than planned` : ''}.{' '}
        <a href="javascript:void 0" onClick={() => setSeed(seed + 1)}>Draw a new set</a>
      </p>
      <Quiz
        key={seed}
        questions={qs}
        title={`${def.title}: result`}
        onFinish={(a) => update((st) => { (st.exams[id] ??= []).push(a); })}
      />
    </div>
  );
}

export function Sources() {
  const groups = new Map<string, { id: string; title: string; url: string }[]>();
  for (const src of Object.values(sources as Record<string, { id: string; title: string; url: string; section: string }>)) {
    if (!groups.has(src.section)) groups.set(src.section, []);
    groups.get(src.section)!.push(src);
  }
  return (
    <div class="page narrow">
      <h1>Sources</h1>
      <p class="ui muted">
        Every source cited in the course, all accessed 2026-10-23. Generated from research/sources.md, which also records the
        precedence rule for conflicting numbers and each source's verification status.
      </p>
      {[...groups].map(([section, list]) => (
        <section>
          <h2>{section}</h2>
          <ul class="src-list" style="font-family: var(--font-ui); font-size: var(--fs-small); gap: .55rem">
            {list.map((s) => (
              <li>
                <a href={s.url} target="_blank" rel="noreferrer">
                  <span class="src-id">{s.id}</span>
                  <span class="src-title">{s.title}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

function download(name: string, text: string, type: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const stamp = () => new Date().toISOString().slice(0, 10);
const tsv = (s: string) => s.replace(/[\t\n\r]+/g, ' ');

export function Data() {
  const s = useStore();
  const [msg, setMsg] = useState('');

  function importFile(e: Event) {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    f.text().then((t) => {
      try {
        const next = JSON.parse(t) as State;
        if (next.version !== 1) throw new Error('version');
        replaceState(next);
        setMsg(`Imported ${f.name}.`);
      } catch {
        setMsg(`${f.name} is not a TPU Deep Dive export. Nothing was changed.`);
      }
    });
  }

  const cards = allCards();
  return (
    <div class="page narrow">
      <h1>Your data</h1>
      <p class="ui muted">
        Progress, quiz attempts, teach-back attempts and flashcard schedules are stored in this browser only (localStorage key
        tpudd:v1). Export to move them to another browser or keep a backup.
      </p>
      <h2>Progress</h2>
      <div class="btn-row">
        <button class="btn" onClick={() => download(`tpu-deep-dive-progress-${stamp()}.json`, JSON.stringify(getState(), null, 1), 'application/json')}>
          Export progress
        </button>
        <label class="btn">
          Import progress
          <input type="file" accept="application/json" class="visually-hidden" onChange={importFile} />
        </label>
      </div>
      <h2>Flashcards</h2>
      <p class="ui muted">
        {cards.length} cards. The Anki file is tab-separated (front, back, tags) with HTML fields; in Anki choose File, Import, and
        tick "Allow HTML in fields". Formulas are exported as rendered KaTeX HTML, so add the KaTeX stylesheet to the Anki card
        template or expect raw markup.
      </p>
      <div class="btn-row">
        <button
          class="btn"
          onClick={() =>
            download(
              `tpu-deep-dive-cards-${stamp()}.txt`,
              '#separator:tab\n#html:true\n#tags column:3\n' + cards.map((c) => `${tsv(c.front)}\t${tsv(c.back)}\ttpu-deep-dive ${c.chapter}`).join('\n'),
              'text/tab-separated-values',
            )
          }
        >
          Export for Anki
        </button>
        <button
          class="btn"
          onClick={() =>
            download(
              `tpu-deep-dive-cards-${stamp()}.json`,
              JSON.stringify(cards.map((c) => ({ ...c, schedule: s.cards[c.id] ?? null })), null, 1),
              'application/json',
            )
          }
        >
          Export with schedules (JSON)
        </button>
      </div>
      <h2>Reset</h2>
      <div class="btn-row">
        <button
          class="btn"
          onClick={() => {
            if (msg === 'confirm-reset') {
              resetState();
              setMsg('All progress was erased.');
            } else setMsg('confirm-reset');
          }}
        >
          {msg === 'confirm-reset' ? 'Erase everything: press again to confirm' : 'Erase all progress'}
        </button>
      </div>
      {msg && msg !== 'confirm-reset' ? <p class="ui" role="status">{msg}</p> : null}
    </div>
  );
}
