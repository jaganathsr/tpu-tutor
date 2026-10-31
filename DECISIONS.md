# Decisions

Calls made without asking, per the brief. Newest last.

1. **Repo initialised with git** in `tpuuniversity/` (it was an empty, non-git directory) so chapters can be committed one at a time.
2. **Research done by five parallel readers**, each writing raw notes to `research/notes/`: Scaling Book ch. 1–6, Scaling Book ch. 7–12, TPU hardware/specs, TPU software stack, GPU/Trainium comparisons + last-12-months changes. `sources.md` and `syllabus.md` are synthesised from those notes; the notes are kept as an audit trail.
3. **Local resources not indexed.** The brief's path is the literal placeholder `[path/to/my/resources]`. Flagged in the syllabus review; will index once a real path is given.
4. **Reference chip is TPU7x (Ironwood).** It is GA and fully documented. Book problems keep their original chip (v5e/v5p) so answers can be checked; each chapter re-runs the key example on Ironwood.
5. **TPU 8 BF16/FP8 peaks are taken from the JAX Pallas hardware table** (`docs.jax.dev/.../pallas/tpu/hardware.html`, generated from libtpu metadata) and tagged "provisional". Google's published TPU 8 spec gives FP4 only.
6. **Number precedence:** Cloud docs > Google papers > Google blogs > JAX docs > Scaling Book > press. Written out in `research/sources.md`.
7. **ICI bandwidth convention:** `W_ici` = bandwidth one chip pushes into one torus axis in a bidirectional ring = per-link bidirectional figure (the book's convention). Cloud docs' per-chip totals are converted explicitly.
8. **All spec numbers live in `scripts/specs.py`** with source IDs; chapter scripts import from it. Re-running for a new chip means adding one entry.
9. **TorchTPU in Ch 9 (Day 5) with a forward reference from Ch 8**, because vllm-torchtpu is now the recommended PyTorch serving path (research finding, 2026-10-20).
10. **Interconnect gets its own chapter** (Ch 3); transformer accounting is Ch 4. Trainium appears as callouts plus one comparison reference page rather than a chapter.
11. **No local resources** (user, 2026-10-23). Section 6 of `sources.md` stays empty.
12. **Show all precisions for every chip** (user, 2026-10-23), including the TPU 8 BF16/FP8/INT8 figures from the JAX hardware table, tagged "provisional".
13. **Syllabus approved as written** (user, 2026-10-23).
14. **Stack: Vite + Preact + MDX**, no backend. Chapters are MDX (`content/chapters/<id>/index.mdx`), quizzes and flashcards are YAML beside them. Markdown and KaTeX inside the YAML are rendered at build time by a small Vite plugin, so the browser never parses math.
15. **Hash routing and `base: './'`** so the static build works from any GitHub Pages path with no 404 fallback. `.github/workflows/pages.yml` deploys on push to `main`.
16. **Type:** Source Serif 4 (body, 18px, optical sizes), IBM Plex Sans (UI and headings), IBM Plex Mono (code and aligned numbers), all self-hosted via Fontsource so the site works offline. Labels are sentence case with light tracking, not all caps.
17. **Color:** neutral paper `#F6F6F3`, ink `#1B1D20`, one cobalt accent `#2C4E9E` for links and active states, sage callouts for GPU analogies and sand for "where the analogy breaks". All text pairs are checked at ≥7:1 in both modes by `scripts/check-contrast.mjs`. Chart series are three colors validated for color-vision deficiency in both modes.
18. **The one bold element** is the home-page course map, drawn as a 5×2 systolic grid (days are columns, blocks are rows, with arrows for data paths) that doubles as the progress tracker.
19. **Spaced repetition is SM-2** with Again/Hard/Good/Easy mapped to quality 1/3/4/5; "Again" re-queues the card within the session. 25 new cards per session.
20. **Free-response questions are self-graded** (covered it / partly / missed it = 1 / 0.5 / 0) after the model answer is shown.
21. **Source titles are generated** from `research/sources.md` by `scripts/build-sources.mjs`; `<Cite id="..." />` in MDX resolves against it and the content linter fails on unknown IDs.
22. **v5e HBM bandwidth is taken as 819 GB/s** (J-hw and SB1 use 8.2e11; 2 stacks of HBM2e at 3.2 Gb/s per pin give 819.2 GB/s). The v5e docs page prints "800 GiBps", which would be 859 GB/s and is most likely a unit slip.
23. **Chapter 11, "TPU and GPU architecture, side by side"** (user request, 2026-10-23). It is a synthesis chapter after Day 5, drawn as a full-width row under the course map, and is included in the final exam. It absorbs the comparison reference page planned earlier. Trainium appears as a third column where the comparison is informative.
24. **Review slides are built into the site** (user request): each chapter has a `slides.mdx` of `<Slide>` blocks, written alongside the chapter. `#/slides/all/0` is the whole-course deck, and it prints at one slide per page for a PDF. I chose this over a separate presentation file so the slides live in the repo, publish with the site and can be edited like the chapters.
25. **Brighter palette** (user request): vivid blue accent `#1452D9` (filled `#1C64F2`), saturated green and orange callout rules, bright yellow flag tags, and one hue per day (blue, teal, orange, pink, violet; green for synthesis) on the map, chapter headers, worked examples and slides. Body text stays ≥7:1. Links and filled buttons were relaxed from 7:1 to 4.5:1 to allow the brighter blue. The header stripe uses hard colour stops, not a blend.
