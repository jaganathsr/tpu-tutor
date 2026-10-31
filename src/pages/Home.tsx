import { chapters, courseInfo, examDefs } from '../lib/content';
import { PASS, chapterProgress, dueCount, newCardCount, totals } from '../lib/progress';
import { useStore } from '../lib/storage';
import { href } from '../lib/router';
import { duration } from '../lib/format';

export function Home() {
  const s = useStore();
  const t = totals(s);
  const due = dueCount(s);
  const fresh = newCardCount(s);
  const next = chapters.find((c) => c.written && !chapterProgress(s, c.id).read) ?? chapters.find((c) => c.written);

  return (
    <div class="page">
      <header class="home-head">
        <h1>{courseInfo.title}</h1>
        <p>{courseInfo.summary}</p>
      </header>

      <div class="summary-line">
        <span><b>{t.read}</b> of <b>{chapters.length}</b> chapters read</span>
        <span>Quiz average <b>{t.quizAvg == null ? 'none yet' : `${Math.round(t.quizAvg * 100)}%`}</b></span>
        <span>Time spent <b>{duration(t.timeMs)}</b></span>
        <span><b>{due}</b> cards due, <b>{fresh}</b> new</span>
      </div>

      <section class="map" aria-label="Course map: five days, two blocks each">
        <div class="map-rowhead corner" />
        {courseInfo.days.map((d) => (
          <div class="map-colhead" style={{ '--dayc': `var(--day${d.day})` }}>
            <strong>Day {d.day}</strong>
            <br />
            {d.theme}
          </div>
        ))}
        {[1, 2].map((block) => (
          <>
            <div class="map-rowhead">{block === 1 ? 'First block' : 'Second block'}</div>
            {courseInfo.days.map((d) => {
              const c = chapters.find((x) => x.day === d.day && x.block === block)!;
              const p = chapterProgress(s, c.id);
              const segs = [p.read, p.quizDone, p.teachDone];
              return (
                <a
                  style={{ '--dayc': `var(--day${d.day})` }}
                  class={`cell ${c.written ? '' : 'unwritten'} ${d.day === 5 ? 'last-col' : ''}`}
                  href={href(`/ch/${c.id}`)}
                  aria-label={`Chapter ${c.number}: ${c.title}. ${c.written ? '' : 'Not written yet.'}`}
                >
                  <span class="cell-num">
                    <span class="cell-day">Day {c.day}, </span>Chapter {c.number}
                  </span>
                  <span class="cell-title">{c.title}</span>
                  <span class="cell-meta">
                    {c.written ? (
                      <>
                        <span>Quiz {p.quizBest == null ? 'not taken' : `best ${Math.round(p.quizBest * 100)}%`}</span>
                        <span>{p.timeMs ? duration(p.timeMs) : `${c.minutes} min planned`}</span>
                      </>
                    ) : (
                      <span>Not written yet</span>
                    )}
                  </span>
                  <span class="cell-progress" title="Read, quiz passed, teach-back saved">
                    {segs.map((d) => (
                      <span class={d ? 'done' : ''} />
                    ))}
                  </span>
                </a>
              );
            })}
          </>
        ))}
        {chapters
          .filter((c) => c.day > courseInfo.days.length)
          .map((c) => {
            const p = chapterProgress(s, c.id);
            return (
              <>
                <div class="map-rowhead">Synthesis</div>
                <a class={`cell synthesis ${c.written ? '' : 'unwritten'}`} style={{ '--dayc': 'var(--day6)' }} href={href(`/ch/${c.id}`)}>
                  <span class="cell-num">Chapter {c.number}, after Day 5</span>
                  <span class="cell-title">{c.title}</span>
                  <span class="cell-meta">
                    {c.written ? `Quiz ${p.quizBest == null ? 'not taken' : `best ${Math.round(p.quizBest * 100)}%`}` : 'Not written yet'}
                  </span>
                  <span class="cell-progress">
                    {[p.read, p.quizDone, p.teachDone].map((x) => <span class={x ? 'done' : ''} />)}
                  </span>
                </a>
              </>
            );
          })}
      </section>

      <div class="btn-row">
        {next ? (
          <a class="btn primary" href={href(`/ch/${next.id}`)}>
            Continue with chapter {next.number}
          </a>
        ) : null}
        <a class="btn" href={href('/review')}>
          Review {due} due {due === 1 ? 'card' : 'cards'}
        </a>
      </div>
      <p class="ui muted">
        Each cell has three marks: chapter read, quiz passed at {PASS * 100}% or more, teach-back saved.
      </p>

      <div class="home-sections">
        <section>
          <h2>Exams</h2>
          <ul class="plain-list">
            {examDefs.map((e) => {
              const att = s.exams[e.id] ?? [];
              const best = att.reduce((m, a) => Math.max(m, a.score / a.total), 0);
              return (
                <li>
                  <a href={href(`/exam/${e.id}`)}>{e.title}</a>
                  <span class="meta">{att.length ? `best ${Math.round(best * 100)}%, ${att.length} attempts` : `${e.count} questions`}</span>
                </li>
              );
            })}
          </ul>
        </section>
        <section>
          <h2>Reference</h2>
          <ul class="plain-list">
            <li><a href={href('/sources')}>Sources</a><span class="meta">every citation, with access date</span></li>
            <li><a href={href('/data')}>Your data</a><span class="meta">export, import, flashcard export</span></li>
          </ul>
        </section>
      </div>
    </div>
  );
}
