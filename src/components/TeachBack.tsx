// Teach-back: objectives as a checklist plus a written explanation. Each saved
// attempt is kept so earlier and later explanations can be compared.
import { useState } from 'preact/hooks';
import type { ChapterMeta } from '../lib/content';
import { chapterState, mutChapter, update, useStore } from '../lib/storage';

const fmtDate = (t: number) =>
  new Date(t).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

export function TeachBack({ ch }: { ch: ChapterMeta }) {
  const s = useStore();
  const cs = chapterState(s, ch.id);
  const last = cs.teach[cs.teach.length - 1];
  const [checked, setChecked] = useState<Set<string>>(new Set(last?.checked ?? []));
  const [compare, setCompare] = useState<number | null>(null);

  const draft = cs.teachDraft ?? '';
  const setDraft = (v: string) => update((st) => { mutChapter(st, ch.id).teachDraft = v; });

  function toggle(i: number) {
    const k = String(i);
    const next = new Set(checked);
    next.has(k) ? next.delete(k) : next.add(k);
    setChecked(next);
  }

  function save() {
    if (!draft.trim()) return;
    update((st) => {
      const c = mutChapter(st, ch.id);
      c.teach.push({ at: Date.now(), checked: [...checked], text: draft.trim() });
      c.teachDraft = '';
    });
  }

  const cmp = compare != null ? cs.teach[compare] : null;

  return (
    <div>
      <p class="ui muted">
        Tick each objective you can deliver out loud without notes. Then write the one-paragraph explanation you would give a
        room of GPU engineers. Saved attempts stay below, so you can compare them over the week.
      </p>
      <ul class="objectives">
        {ch.objectives.map((o, i) => (
          <li>
            <label>
              <input type="checkbox" checked={checked.has(String(i))} onChange={() => toggle(i)} />
              <span>{o}</span>
            </label>
          </li>
        ))}
      </ul>
      {ch.teachback.length ? (
        <>
          <h3>Prompts to answer out loud</h3>
          <ol class="prompts">
            {ch.teachback.map((p) => (
              <li>{p}</li>
            ))}
          </ol>
        </>
      ) : null}
      <label class="field" style="margin-top:1.4rem">
        <span>Your explanation</span>
        <textarea value={draft} onInput={(e) => setDraft((e.target as HTMLTextAreaElement).value)} />
      </label>
      <div class="btn-row">
        <button class="btn primary" onClick={save} disabled={!draft.trim()}>
          Save attempt
        </button>
        <span class="ui muted">
          {checked.size} of {ch.objectives.length} objectives ticked. Drafts save as you type.
        </span>
      </div>

      {cs.teach.length ? (
        <section class="attempts">
          <h3>Saved attempts</h3>
          {cmp && last ? (
            <div class="compare">
              <div>
                <div class="attempt-meta">{fmtDate(cmp.at)}, {cmp.checked.length} of {ch.objectives.length} ticked</div>
                <p>{cmp.text}</p>
              </div>
              <div>
                <div class="attempt-meta">Latest: {fmtDate(last.at)}, {last.checked.length} of {ch.objectives.length} ticked</div>
                <p>{last.text}</p>
              </div>
            </div>
          ) : null}
          {[...cs.teach].reverse().map((t, ri) => {
            const idx = cs.teach.length - 1 - ri;
            return (
              <div class="attempt">
                <div class="attempt-meta">
                  {fmtDate(t.at)}, {t.checked.length} of {ch.objectives.length} objectives ticked
                  {idx < cs.teach.length - 1 ? (
                    <>
                      {'. '}
                      <a href="javascript:void 0" onClick={() => setCompare(compare === idx ? null : idx)}>
                        {compare === idx ? 'Hide comparison' : 'Compare with latest'}
                      </a>
                    </>
                  ) : null}
                </div>
                <p>{t.text}</p>
              </div>
            );
          })}
        </section>
      ) : null}
    </div>
  );
}
