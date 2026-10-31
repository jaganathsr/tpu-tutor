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
