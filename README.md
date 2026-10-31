# TPU Deep Dive

A self-paced course on TPU hardware and software for engineers who already know GPUs. It has 11 chapters: five days of two ~55-minute blocks each, plus a synthesis chapter comparing TPU, GPU and Trainium. Each chapter has prose with inline citations, interactive explainers, a quiz, spaced-repetition flashcards, a teach-back checklist and review slides. There are day exams, a 30-question final, and a whole-course slide deck.

It is a static site (Vite + Preact + MDX) with no backend. Progress is stored in your browser.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/
npm run preview    # serve dist/ on http://localhost:4173
```

Node 20 or newer. `npm run dev` and `npm run build` regenerate `src/generated/sources.json` and `src/generated/chapters-meta.json` first.

`npm run build` does three things:

1. Builds the client app.
2. Builds a server bundle (`src/entry-server.tsx`).
3. Runs `scripts/prerender.mjs`, which writes a static HTML file for every route (`dist/ch/ch07/index.html` and so on), plus `404.html` and `sitemap.xml`.

Two environment variables control URLs:

| Variable | Default | Pages value (set in the workflow) |
|---|---|---|
| `BASE_PATH` | `/` | `/` |
| `SITE_URL` | `https://learntpu.dev` | same |
| `UMAMI_WEBSITE_ID` | unset (no analytics) | repository variable `UMAMI_WEBSITE_ID` |
| `UMAMI_SRC` | `https://cloud.umami.is/script.js` | set only for self-hosted Umami |

## Publish on GitHub Pages

1. Push the repository to GitHub, on the `main` branch.
2. In the repository settings, under Pages, set Source to **GitHub Actions**.
3. `.github/workflows/pages.yml` runs `npm run check` and `npm run build` on every push and publishes `dist/`.

Pages are real paths (`https://learntpu.dev/ch/ch07/`), each prerendered to static HTML. Routes that are not prerendered, such as individual slide numbers, are served by `404.html`, which boots the app and renders from the URL. Old `#/…` links redirect to the matching path. The custom domain is set by `public/CNAME` and in the repository's Pages settings. To serve from a project path instead, such as `https://user.github.io/repo/`, set `BASE_PATH=/repo/` and `SITE_URL` in `.github/workflows/pages.yml`.

**Analytics.** Set a repository variable `UMAMI_WEBSITE_ID` (Settings → Secrets and variables → Actions → Variables) and re-run the workflow. The Umami script is then added to every page, and the Your data page says that page views are counted. Without the variable, no analytics script is included.

## Using the course

| Where | What |
|---|---|
| Course (home) | The course map: days are columns, blocks are rows. Each cell shows three marks: read, quiz passed at 80%, teach-back saved. |
| Chapter → Read | Prose, worked examples, interactives, and a source list with section anchors (left column on wide screens). |
| Chapter → Quiz | 12 questions: recall, numeric (checked within a tolerance), spot-the-bottleneck, and one explain-to-a-GPU-engineer with a model answer. |
| Chapter → Flashcards | About 20 cards, scheduled with SM-2. |
| Chapter → Teach-back | Objectives as a checklist plus a written explanation; every saved attempt is kept for comparison. |
| Review | All due flashcards across chapters. |
| Slides | The whole course as one deck (`/slides/all/0/`), or one chapter (`/slides/ch05/0/`). |
| Exams | Five day exams (10 questions plus 2 from earlier days) and a 30-question final. Each attempt draws a new set with minimum numbers of numeric, scenario and written questions. |

Keyboard:

- **Quiz:** 1–9 choose an option, Enter checks and advances, Ctrl+Enter submits a written answer, 1–3 self-grade it.
- **Flashcards:** Space flips, 1–4 grade.
- **Slides:** arrow keys, Space, Home/End, F for full screen. "Print or save as PDF" prints one slide per page.

## Export flashcards and progress

Open **Your data** (footer link or `/data/`):

- **Export for Anki:** a tab-separated file with `front`, `back` and `tags` columns and HTML fields. In Anki use File → Import and tick "Allow HTML in fields". Formulas are exported as rendered KaTeX HTML, so add the KaTeX stylesheet to the card template or expect raw markup.
- **Export with schedules (JSON):** every card plus its SM-2 state.
- **Export / Import progress:** the whole localStorage state (key `tpudd:v1`) as JSON, to move between browsers or keep a backup.

## Add or edit a chapter

A chapter is a directory, `content/chapters/<id>/`, listed in `content/course.yaml`:

```yaml
- { id: ch12, number: 12, day: 6, block: 1, minutes: 55, title: "New chapter" }
```

Days 1–5 appear as map columns; a `day` above 5 appears in the full-width synthesis row.

| File | Contents |
|---|---|
| `index.mdx` | Frontmatter (`description` of at most 160 characters for search results, `objectives`, `teachback`, `sources`, optional `status: draft`) and the chapter body in Markdown. KaTeX math uses `$...$` and `$$...$$`, with `\tag{n.m}` for numbering. |
| `quiz.yaml` | 8–12 questions. Types: `mcq` and `scenario` (each option has `text`, `why`, and exactly one `correct: true`), `numeric` (`answer`, `tolerance` relative, `unit`), `free` (`model_answer`). Optional `explanation`, `source` (e.g. `"SB1 #matrix-multiplication"`) and `adapted_from`. |
| `cards.yaml` | 15–25 cards with `id`, `front`, `back` and `source`. Markdown and math are allowed and rendered at build time. |
| `slides.mdx` | `<Slide title="...">...</Slide>` blocks. Use `<Cols>` for two columns, `<Big n="313">label</Big>` for a headline number, and plain tables and lists. Keep interactives out of slides; use `<ChipTable>` and static diagrams. |
| `dumps/*.txt` | Optional generated listings, shown with `<Listing src="ch09/1_jaxpr.txt" title="..." />`. |

Components available in `index.mdx` without imports (`src/components/mdx.tsx`):

- `<Cite id="SB1" at="#matrix-multiplication" s="Q2" />`: an inline citation; the ID must exist in `research/sources.md`.
- `<Gloss>`: the plain-English line under each formula.
- `<GpuAnalogy>`, `<AnalogyBreaks>`, `<Note>` (margin note), `<Worked title="...">`, `<Wide>` (borrows the margin column).
- `<Unverified />`, `<Provisional />`, `<Derived />`: visible tags.
- `<ChipTable chips="7x H100" cols="bf16 fp8 hbm hbm_bw ai" />`: spec table from `scripts/specs.py`.
- `<Interactive caption={...}>` wrapping any of the interactives:
  - `RooflinePlotter`, `GenerationChart`, `SystolicArray`, `TensorCoreDiagram`, `TorusExplorer`, `ModelCalc`
  - `ShardingSandbox`, `CollectiveViz`, `MeshPlanner`, `BreakevenCalc`, `ServingStackDiagram`
  - `HloDecoder`, `TraceReader`, `CompareExplorer`

`npm run check` lints content against the writing rules: banned phrases, exclamation marks and emoji. It also checks citation IDs, the quiz schema, question and card counts, and that each chapter has at least one question adapted from the Scaling Book. It also checks colour contrast.

## Search engines

What the build does for search:

- **Per-page metadata:** every chapter is a prerendered page with its full text in the HTML, plus a `<title>`, a meta description (the chapter's `description`), a canonical URL, Open Graph and Twitter card tags, and schema.org JSON-LD (`Course` on the home page; `LearningResource` and `BreadcrumbList` on chapters).
- **Noindex app pages:** quiz, flashcard, teach-back, review, exam, slide and data pages are prerendered with `noindex`.
- **Sitemap:** `sitemap.xml` lists the 16 indexable URLs (home, Sources, About, Privacy, Terms and 11 chapters).
- **Share image:** `public/og.png`, rendered from `scripts/og-card.html`; the re-render command is in that file.
- **Code splitting:** each chapter's content is its own JavaScript chunk.

What only you can do:

1. **Submit the site to Google Search Console** (search.google.com/search-console).
   - Add a Domain property for `learntpu.dev`, verified with a DNS TXT record at the registrar, or a URL-prefix property `https://learntpu.dev/`.
   - Verify it with the HTML-tag method: add the `<meta name="google-site-verification" …>` tag inside the `<!--head:start-->` block of `index.html`, push, then click Verify.
   - Submit `https://learntpu.dev/sitemap.xml`.
2. **Do the same in Bing Webmaster Tools.** It can import the Search Console property.
3. `robots.txt` is generated at the domain root and points to the sitemap. It is only written when `SITE_URL` has no path.
4. **Links bring the most traffic:** the GitHub repository's description and website field, topics, and posts linking to individual chapters.

## Sources and numbers

- `research/sources.md` lists every citable source with an ID, verification status and the precedence rule for conflicting numbers. Add a row there to make a new `<Cite id>` valid. `research/notes/` holds the raw reading notes; `research/syllabus.md` holds the plan.
- `scripts/specs.py` holds every chip spec with its source. To add a chip, add an entry, then run `python3 scripts/specs.py --json` to regenerate `src/generated/chips.json`. Every interactive picks it up.
- `scripts/model.py` (mirrored in `src/lib/model.ts`) holds transformer accounting and model presets.
- `scripts/chNN_*.py` recompute every number used in each chapter. Run any of them with `python3 scripts/<name>.py` from the repository root.
- Two scripts need JAX:
  - `scripts/ch09_dump.py` generates the compiler listings in Chapter 9.
  - `scripts/ch10_pallas_matmul.py` tests the Pallas kernel in interpret mode.

  Run them with a JAX environment, for example:

  ```bash
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python scripts/ch09_dump.py
  ```

- `python3 scripts/check_urls.py` re-checks every source URL and writes `research/url_check.txt`.

`DECISIONS.md` records the calls made while building the course.

## Tests

```bash
npm run check                          # content lint and contrast
npm run build && npm run preview &     # then:
npm run test:e2e                       # keyboard-driven quiz, flashcard and teach-back flows in Chrome
npx vite preview --port 4174 &
BASE=http://localhost:4174 npm run test:routing   # or BASE=https://learntpu.dev   # paths, links, back button, legacy #/ redirect
```

`test:e2e` uses `playwright-core` with the system Chrome (`CHROME=/path/to/chrome` to override).
