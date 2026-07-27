# HUMAN_TODO — WealthLens Analyst

Items only Chris can close. Agents may add items and tick one off **only** when
completion is directly verified; never infer an approval, a decision, or a
subjective confirmation.

## Decisions

- [ ] **Confirm the local `sensitive_data` posture.** The estate registry recommends the
      `sensitive_data` overlay for this repo because of the gitignored root `.env`.
      `.agent-harness/tier.json` sets it **false** on purpose: the floor's `sensitive_data`
      overlay denies pushes to public remotes, and this remote is public by your own
      2026-07-26 decision — the flag would make the repo unpushable by agents while
      protecting nothing `.gitignore` does not already cover (`.env` is ignored; measured
      2026-07-27: no secret is tracked in the tree). Confirm this reading, or say the word
      and the overlay goes on and agents stop pushing here.
- [ ] **Ratify tier 2.** Bootstrapped 2026-07-27 under law 9 as T2 daily driver
      (`.agent-harness/tier.json`), `push: free` / `merge: free`. Re-review for T3 when
      H1-30 puts a live URL in front of real users.
- [ ] **`main` has no branch protection** (measured 2026-07-27,
      `gh api repos/Chris0Jeky/wealthlens-analyst/branches/main/protection` → 404).
      Decide whether to require the CI check before merge. Squash-merge is already
      disabled repo-side, which matches estate policy.

## Product blockers (from `tasks/hero1-backlog.md` — these gate M5/M6)

- [ ] **H1-02 / H1-24 — golden answers.** 20 golden records exist, **0 reviewed, 20 DRAFT**
      (measured: `python evals/checks/deterministic.py`). Agents must never write these:
      a test (`tests/test_golden_draft_guard.py`) fails if a DRAFT record carries an answer.
      RAGAS (H1-25), the eval report (H1-26) and the abstention-gate calibration (H1-22's
      calibration note) are all blocked on ≥50 reviewed pairs.
- [ ] **`COHERE_API_KEY`** — H1-16's reranker stays unbuilt (flag OFF, `NotImplementedError`)
      until you supply a key; flag-on cannot be honestly verified without it.
- [ ] **H1-30 hosting** — provision the Hetzner CAX21 decided in ADR 0003 D3, deploy, run
      Alembic + ingest. `/healthz` green on the public URL is the gate.
- [ ] **H1-32 demo sends** — the 10 named recipients and the send log are maintained
      privately by you; agents must not contact anyone.
- [ ] **ADR 0003 D3 follow-on: Langfuse placement/sizing** on the chosen box (H1-28).

## Notes for agents

Nothing here is an excuse to stall. Everything not listed above — code, tests, migrations,
evals plumbing, docs, dependency PRs — is agent work under the T2 gate.
