import { useRoute, href } from './lib/router';
import { update, useStore } from './lib/storage';
import { dueCount } from './lib/progress';
import { Home } from './pages/Home';
import { ChapterPage } from './pages/Chapter';
import { Data, Exam, Exams, Review, Sources } from './pages/Other';
import { SlidesPage } from './pages/Slides';

function ThemeToggle() {
  const s = useStore();
  const cur = s.prefs.theme ?? (document.documentElement.dataset.theme as 'light' | 'dark' | undefined) ?? 'system';
  const next = cur === 'system' ? 'light' : cur === 'light' ? 'dark' : 'system';
  return (
    <button
      class="theme-toggle"
      onClick={() => {
        update((st) => {
          if (next === 'system') delete st.prefs.theme;
          else st.prefs.theme = next;
        });
        if (next === 'system') delete document.documentElement.dataset.theme;
        else document.documentElement.dataset.theme = next;
      }}
      aria-label={`Colour theme: ${cur}. Switch to ${next}.`}
    >
      Theme: {cur}
    </button>
  );
}

export function App() {
  const path = useRoute();
  const s = useStore();
  const parts = path.split('/').filter(Boolean);
  const section = parts[0] ?? '';

  let page;
  if (section === '') page = <Home />;
  else if (section === 'ch' && parts[1]) page = <ChapterPage id={parts[1]} tab={(parts[2] as any) ?? 'read'} />;
  else if (section === 'slides') page = <SlidesPage scope={parts[1] ?? 'all'} n={Number(parts[2] ?? 0) || 0} />;
  else if (section === 'review') page = <Review />;
  else if (section === 'exams') page = <Exams />;
  else if (section === 'exam' && parts[1]) page = <Exam id={parts[1]} />;
  else if (section === 'sources') page = <Sources />;
  else if (section === 'data') page = <Data />;
  else
    page = (
      <div class="page narrow">
        <p class="empty">
          No page at {path}. <a href={href('/')}>Go to the course map</a>.
        </p>
      </div>
    );

  const due = dueCount(s);
  const cur = (name: string) => (section === name || (name === '' && section === 'ch') ? 'page' : undefined);

  return (
    <>
      <header class="site-header">
        <div class="inner">
          <a class="site-title" href={href('/')}>
            TPU Deep Dive
          </a>
          <nav class="site-nav" aria-label="Main">
            <a href={href('/')} aria-current={cur('')}>Course</a>
            <a href={href('/review')} aria-current={cur('review')}>
              Review <span class="due-count">{due ? `(${due} due)` : ''}</span>
            </a>
            <a href={href('/slides/all/0')} aria-current={cur('slides')}>Slides</a>
            <a href={href('/exams')} aria-current={section === 'exams' || section === 'exam' ? 'page' : undefined}>Exams</a>
            <a href={href('/sources')} aria-current={cur('sources')}>Sources</a>
            <ThemeToggle />
          </nav>
        </div>
      </header>
      {page}
      <footer class="site-footer">
        <div class="inner">
          <span>Sources accessed 2026-10-23. Numbers marked provisional, derived or unverified are explained where they appear.</span>
          <a href={href('/data')}>Your data</a>
        </div>
      </footer>
    </>
  );
}
