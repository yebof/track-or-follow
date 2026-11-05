# v2 exploratory spec: portfolio construction, horizon, event-driven news (2026-09-29)

**Status: EXPLORATORY, post hoc.** Written after the registered W_clean results (PD1, H3 fallback, E2, E4) were
known, to test the three fixes the E4/E2 diagnosis suggests. Nothing here is confirmatory, nothing enters the Holm
family, `results/trials.jsonl` or `results/analysis/`. Outputs go to `results/v2/`. Cost: $0 (replays of existing
judgment tables and prices; no model call). Registered code in `src/jevtrader/` is imported read-only and not edited
(adding files there would change `code_hash`), so all v2 code lives in `experiments/v2/`.

**Protocol (fixed before any W_clean v2 number is computed).**
1. Development and plumbing checks on `W_pre_slice` (2023-04-07..2024-05-31) only. Code features exist for U100;
   news judgments exist for S30 only and sit before the emulator's documented knowledge cutoff, so W_pre news
   numbers are recall-contaminated and used for plumbing only, never to choose a variant.
2. The variant grid and the one primary per item below are fixed in this file before the W_clean run; the file's
   SHA-256 is printed into every v2 output.
3. One evaluation pass on `W_clean`; the news item is also reported on `W_release` (post-release, less recall risk).
4. Every variant is reported (no selection); the number of variants is stated next to every Sharpe.

## Item 1 + 2: portfolio construction and horizon (trusted code features only)

Scores (cross-sectional, on U_t, each feature rank-normalised to [-1, 1] per week; missing = 0):
- `MOM`: mom_12_1.
- `UAW`: mean of the 14 trusted *code* features signed by `configs/learning_v1.yaml: uaw_signs` (signs fixed on
  2026-09-28 before unlock); no outcome learning.
- `FM4`: Fama-MacBeth: weekly cross-sectional OLS of the 4-week relative return (r4 - m4) on the 14 ranked code
  features, using only labels with `available_at4 < c_t`; slope = mean of the trailing 104 matured weeks, shrunk
  by 0.5 toward 0; UAW until 26 matured weeks exist. Training rows may come from W_pre (past data).
- `FM1`: same with the 1-week label (r - m) and `available_at`.

Books (all long-only, fully invested, no inverse-vol, no cash buffer, 15 bps per side):
- `TOP`: top-20 by score, equal weight.
- `BUF`: top-20 entry, a held name is kept while its rank <= 40; fill to 20 by rank; industry (FF12) <= 6 names;
  equal weight at rebalance.
- `TILT`: all U_t names, w_i = (1 + z_i)/N with z_i the score's rank in [-1, 1] (weights in [0, 2/N]).

Rebalance: every week (`R1`) or every 4th decision week (`R4`, weights drift in between).
Grid: 4 scores x 3 books x 2 rebalances = 24 variants.
**Primary (item 1+2): `UAW-BUF-R4`.** Benchmarks: EW-U100 (weekly), the registered D-sized and D-sel-EW.

## Item 4: event-driven news overlay

Signal: kept, C2-eligible N1 items (same frame rule as H2). Item score e = E[effect] - 2 in [-2, 2]. For stock i and
session k, s_{i,k} = sum of e over items with created_at in (cutoff_{k-1}, cutoff_k].
- Cutoff `A` (pre-open): 09:25 ET on session k; trade at open(k).
- Cutoff `B` (prior close): 16:00 ET on session k-1; trade at open(k). (B never uses overnight news for day k.)
Event: positive if s >= +1, negative if s <= -1; active for H sessions from the entry open (a new event restarts
the clock; positive and negative in the same window net by the latest event).
Overlay on a base book: at each open, multiplier m_i = 2 (positive active), 0 (negative active), 1 otherwise;
target = base drifted weight x m, renormalised to 1. Only names whose multiplier changes are re-sized (the rest keep
their drifted weights up to the renormalisation), so the overlay does not pay daily drift rebalancing.
Grid: cutoff {A, B} x H {1, 2, 5} x base {EW-U100 weekly, item-1 primary} = 12 variants.
**Primary (item 4): cutoff A, H = 2, base EW-U100.** Diagnostic: item-level Spearman of e with the open-to-open
market-relative return from the entry open over 1, 2 and 5 sessions, by cutoff.

## Metrics
Annualised excess return over DFF, vol, Sharpe with the registered studentized block-bootstrap CI (block 5), max
drawdown, one-way turnover per year, cost drag, active return vs EW-U100 with tracking error and IR. Per-variant
count reported; no p-values.

## Known limits
Opening-auction fills at the adjusted open with 15 bps are optimistic for cutoff A near the open. W_clean items
before 2025-04-14 may carry emulator recall (E3); hence the W_release row. One historical path, 115 weeks.

## Addendum v2.1 (2026-09-30, before any W_hist number): choosing the deployable book

The W_clean grid above has now been seen (docs/impl/v2_report.md), so W_clean can no longer pick a variant. The
choice therefore moves to a development window that shares no week with W_clean:
- **W_hist** = 2019-06-07..2023-03-31. It uses trusted code features only (`scripts/build_code_features.py`, same code
  as MAIN's), and there is no LLM in it.
- **Rule (mechanical):** among the 24 grid variants with turnover <= 4x/yr on W_hist, pick the one with the highest
  net Sharpe on W_hist; ties within 0.02 go to the lower-turnover one. FM slopes use only labels matured before each
  decision. From here on, the W_clean FM rows train on W_hist + W_pre + past W_clean labels.
- **News risk filter** (the typed judgments' role after item 4): a negative-only overlay on the chosen book
  (m = 0 for s <= -1, cutoff A, H = 5; no positive boost). It is kept iff its net active return vs the chosen book on
  W_release (the least recall-exposed period) is >= 0. This is decided on W_release data, so the result is labelled
  post hoc.
- **Report:** the chosen book on W_hist, W_pre and W_clean, next to EW-U100 and the registered D-sized. The W_clean
  row is a re-look, not an independent test.
