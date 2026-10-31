import { sitePages } from '../lib/content';
import { mdxComponents } from '../components/mdx';

export function SitePage({ slug }: { slug: string }) {
  const p = sitePages[slug];
  if (!p) return <div class="page narrow"><p class="empty">No page called {slug}.</p></div>;
  const { Content } = p;
  const updated = new Date(String(p.updated) + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
  return (
    <div class="page narrow">
      <h1>{p.title}</h1>
      <p class="ui muted">Last updated {updated}</p>
      <article class="prose site-page">
        <Content components={mdxComponents} />
      </article>
    </div>
  );
}
