// Quiz engine, shared by chapter quizzes and exams.
// Keys: 1-9 choose an option, Enter submits and advances, Ctrl+Enter submits a
// written answer, 1-3 self-grade a written answer after the model answer shows.
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks';
import type { Question } from '../lib/content';
import type { QuizAttempt } from '../lib/storage';
import { parseNumber, sig } from '../lib/format';
import { Cite } from './mdx';

const KIND: Record<Question['type'], string> = {
  mcq: 'Recall',
  numeric: 'Derivation: enter a number',
  scenario: 'Spot the bottleneck',
  free: 'Explain to a GPU engineer',
};

const SELF = [
  { k: '1', label: 'Covered it', pts: 1 },
  { k: '2', label: 'Partly', pts: 0.5 },
  { k: '3', label: 'Missed it', pts: 0 },
];

type Answer = { given: string; points: number; done: boolean };

function Html({ html, class: cls }: { html?: string; class?: string }) {
  return html ? <div class={cls} dangerouslySetInnerHTML={{ __html: html }} /> : null;
}

function SourceLine({ q }: { q: Question }) {
  if (!q.source && !q.adapted_from) return null;
  const [id, ...rest] = (q.source ?? '').split(' ');
  return (
    <div class="q-source">
      {q.adapted_from ? <>Adapted from {q.adapted_from}. </> : null}
      {q.source ? (
        <>
          Source: <Cite id={id} at={rest.join(' ') || undefined} />
        </>
      ) : null}
    </div>
  );
}

export function Quiz({ questions, onFinish, title }: { questions: Question[]; onFinish: (a: QuizAttempt) => void; title?: string }) {
  const [i, setI] = useState(0);
  const [answers, setAnswers] = useState<Record<string, Answer>>({});
  const [choice, setChoice] = useState<number | null>(null);
  const [text, setText] = useState('');
  const [finished, setFinished] = useState(false);
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);

  const q = questions[i];
  const a = q ? answers[q.id] : undefined;
  const revealed = Boolean(a);

  useEffect(() => {
    inputRef.current?.focus();
  }, [i]);

  // Reset per-question input in the same update as the move, not in an effect,
  // so text typed right after advancing is never cleared.
  function goTo(n: number) {
    setChoice(null);
    setText('');
    setI(n);
  }

  const record = (ans: Answer) => setAnswers((prev) => ({ ...prev, [q.id]: ans }));

  function submit() {
    if (!q || revealed) return;
    if (q.type === 'mcq' || q.type === 'scenario') {
      if (choice == null) return;
      record({ given: String(choice), points: q.options![choice].correct ? 1 : 0, done: true });
    } else if (q.type === 'numeric') {
      const v = parseNumber(text);
      if (v == null) return;
      const tol = q.tolerance ?? 0.05;
      const ok = Math.abs(v - q.answer!) <= tol * Math.abs(q.answer!);
      record({ given: text, points: ok ? 1 : 0, done: true });
    } else {
      if (!text.trim()) return;
      record({ given: text, points: 0, done: false }); // waits for self-grade
    }
  }

  function selfGrade(pts: number) {
    if (!a || a.done) return;
    record({ ...a, points: pts, done: true });
  }

  function next() {
    if (!a?.done) return;
    if (i + 1 < questions.length) goTo(i + 1);
    else finish();
  }

  function finish() {
    const score = Object.values(answers).reduce((s, x) => s + x.points, 0);
    onFinish({
      at: Date.now(),
      score,
      total: questions.length,
      answers: Object.fromEntries(Object.entries(answers).map(([k, v]) => [k, { correct: v.points, given: v.given }])),
    });
    setFinished(true);
  }

  // One listener for the component's lifetime; it always calls the latest handler,
  // so fast key sequences never hit a stale render.
  const onKeyRef = useRef<(e: KeyboardEvent) => void>(() => {});
  onKeyRef.current = (e: KeyboardEvent) => {
      if (finished || !q) return;
      const inField = (e.target as HTMLElement)?.matches?.('input, textarea, select');
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        revealed ? next() : submit();
        return;
      }
      if (e.key === 'Enter' && !(inField && q.type === 'free' && !revealed)) {
        e.preventDefault(); // also stops a focused button from re-clicking
        revealed ? next() : submit();
        return;
      }
      if (inField && !revealed) return;
      if (/^[1-9]$/.test(e.key)) {
        const n = +e.key - 1;
        if (!revealed && q.options && n < q.options.length) setChoice(n);
        else if (revealed && q.type === 'free' && !a?.done && n < 3) selfGrade(SELF[n].pts);
      }
    };
  useLayoutEffect(() => {
    const onKey = (e: KeyboardEvent) => onKeyRef.current(e);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  if (!questions.length) return <p class="empty">No questions yet.</p>;

  if (finished) {
    const score = Object.values(answers).reduce((s, x) => s + x.points, 0);
    return (
      <div class="quiz">
        <h2 style="margin-top:0">{title ?? 'Result'}</h2>
        <p class="ui">
          Score: <b class="num">{sig(score)}</b> of <span class="num">{questions.length}</span> (
          <span class="num">{Math.round((100 * score) / questions.length)}%</span>).
        </p>
        <table class="score-table">
          <thead>
            <tr><th>Question</th><th>Type</th><th class="num">Points</th></tr>
          </thead>
          <tbody>
            {questions.map((qq, n) => (
              <tr>
                <td>
                  <a href="javascript:void 0" onClick={() => { setFinished(false); goTo(n); }}>
                    {n + 1}
                  </a>
                </td>
                <td>{KIND[qq.type]}</td>
                <td class="num">{answers[qq.id] ? sig(answers[qq.id].points) : '0'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div class="btn-row">
          <button class="btn primary" onClick={() => { setAnswers({}); goTo(0); setFinished(false); }}>
            Start a new attempt
          </button>
        </div>
      </div>
    );
  }

  const isChoice = q.type === 'mcq' || q.type === 'scenario';

  return (
    <div class="quiz">
      <div class="quiz-progress">
        <span>
          Question {i + 1} of {questions.length}
        </span>
        <span class="bar" aria-hidden="true">
          {questions.map((qq, n) => {
            const x = answers[qq.id];
            const cls = x?.done ? (x.points >= 1 ? 'right' : 'wrong') : n === i ? 'cur' : '';
            return <span class={cls} />;
          })}
        </span>
      </div>
      <div class="q-kind">{KIND[q.type]}</div>
      <Html class="q-prompt" html={q.prompt} />

      {isChoice ? (
        <ul class="options">
          {q.options!.map((o, n) => {
            const chosen = revealed ? a!.given === String(n) : choice === n;
            let cls = 'option';
            let verdict = '';
            if (revealed) {
              if (o.correct) { cls += ' is-correct'; verdict = chosen ? 'Correct. Your answer.' : 'Correct answer.'; }
              else if (chosen) { cls += ' is-wrong'; verdict = 'Incorrect. Your answer.'; }
              else { cls += ' not-chosen'; verdict = 'Incorrect.'; }
            }
            return (
              <li>
                <button class={cls} aria-pressed={chosen} disabled={revealed} onClick={() => setChoice(n)}>
                  <span class="key">{n + 1}</span>
                  <span>
                    <span dangerouslySetInnerHTML={{ __html: o.text }} />
                    {revealed ? (
                      <>
                        <span class="verdict">{verdict}</span>
                        <span class="why" dangerouslySetInnerHTML={{ __html: o.why }} />
                      </>
                    ) : null}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}

      {q.type === 'numeric' ? (
        <div class="numeric-row">
          <input
            ref={inputRef as any}
            type="text"
            inputMode="decimal"
            aria-label="Your answer"
            value={revealed ? a!.given : text}
            disabled={revealed}
            onInput={(e) => setText((e.target as HTMLInputElement).value)}
            placeholder="e.g. 1.2e3"
          />
          {q.unit ? <span class="unit">{q.unit}</span> : null}
        </div>
      ) : null}

      {q.type === 'free' ? (
        <textarea
          ref={inputRef as any}
          aria-label="Your explanation"
          value={revealed ? a!.given : text}
          disabled={revealed}
          onInput={(e) => setText((e.target as HTMLTextAreaElement).value)}
          placeholder="Write the explanation you would give out loud."
        />
      ) : null}

      {revealed && q.type === 'numeric' ? (
        <div class={`result ${a!.points ? 'right' : 'wrong'}`}>
          <span class="verdict">{a!.points ? 'Correct.' : 'Incorrect.'}</span> Expected {sig(q.answer!, 4)}
          {q.unit ? ` ${q.unit}` : ''}, accepted within ±{Math.round((q.tolerance ?? 0.05) * 100)}%.
          <Html class="body" html={q.explanation} />
        </div>
      ) : null}

      {revealed && isChoice && q.explanation ? (
        <div class="result">
          <Html class="body" html={q.explanation} />
        </div>
      ) : null}

      {revealed && q.type === 'free' ? (
        <div class="result">
          <span class="verdict">Model answer</span>
          <Html class="body" html={q.model_answer} />
          {!a!.done ? (
            <div class="btn-row">
              <span class="ui muted">Compare with yours:</span>
              {SELF.map((s) => (
                <button class="btn" onClick={() => selfGrade(s.pts)}>
                  {s.label}
                  <kbd>{s.k}</kbd>
                </button>
              ))}
            </div>
          ) : (
            <p class="ui muted" style="margin-top:.6rem">Self-graded: {SELF.find((s) => s.pts === a!.points)?.label}.</p>
          )}
        </div>
      ) : null}

      {revealed ? <SourceLine q={q} /> : null}

      <div class="btn-row">
        {!revealed ? (
          <button class="btn primary" onClick={submit}>
            Check answer<kbd>{q.type === 'free' ? 'Ctrl Enter' : 'Enter'}</kbd>
          </button>
        ) : (
          <button class="btn primary" onClick={next} disabled={!a!.done}>
            {i + 1 < questions.length ? 'Next question' : 'See result'}
            <kbd>Enter</kbd>
          </button>
        )}
        {isChoice && !revealed ? <span class="keys-hint">Press 1–{q.options!.length} to choose.</span> : null}
      </div>
    </div>
  );
}
