# Pre-registration v1: W_clean confirmatory analysis

**Status: final pre-commit text of 2026-09-28, for the commit of 2026-09-28 (brought forward from the 2026-10-02 plan
because all inputs were ready; §17). Not committed. `pipeline/lock.py` stays locked (`results/locks/` does not
exist).** The commit happens before any W_clean outcome is read. After it, only bug fixes are allowed, each logged in
`docs/design/prereg_deviations.md` (design_v1 change control). The decisions P-1..P-8 are closed (§0.2), and the
findings of the pre-commit audit of 2026-09-28 are fixed and registered (§0.3). The commit procedure is §17; this
text does not carry it out.

**File SHA-256: `[PLACEHOLDER: computed at the commit over this file exactly as committed; recorded in
configs/prereg_v1.yaml and printed per §17 step 4, never written back into this file]`**

**Sources.**
- Design: `docs/design/design_v1.md` (§13 in full) and `docs/design/design_v2_delta.md` (§7 items 1–10). Where they
  differ, the delta wins.
- Decisions: `docs/design/decisions.md`, rows of 2026-09-28: the user's $250 total LLM budget, the re-plan, and
  P-1..P-8.
- State of the runs: `docs/impl/freeze_report.md`, `pre_report.md`, `main_report.md`, `gens_report.md`,
  `ng_report.md`, `precommit_paid.md`, `precommit_redteam.md`, `precommit_analysis.md`, `precommit_live.md`, and
  `precommit_report.md` (this finalization).

**Numbers.** Every number below comes from a script output (CLAUDE.md rule 2) and names its source:
- `results/prereg/completeness_audit.json` (audit), from `experiments/prereg/audit_generations.py`;
- `results/prereg/power_prereg.json` (power), from `experiments/prereg/power_prereg.py`;
- `results/prereg/hashes.json` (hashes), from `experiments/prereg/prereg_hashes.py`;
- `results/analysis/CODE_HASHES.json` (code) and `results/analysis/GENERATION_HASHES.json` (generations), from
  `experiments/analysis/code_hashes.py` (`--generations` for the second);
- `results/redteam/precommit_hashes.json` (red-team hashes), from `experiments/redteam/precommit_hashes.py`;
- `results/analysis/budget_snapshot.json` (budget), from `experiments/analysis/budget_snapshot.py`;
- `results/analysis/simulations/null_rates.json` (simulation), from `experiments/analysis/sim_null_rates.py`;
- `results/analysis/h2_frame.json` (frame), from `experiments/analysis/h2_validity.py --frame-only`;
- each generation's `MANIFEST.json`, `projection.json` and `validation.json`;
- plus the result files these name.

The values are those of 2026-09-28. At the commit the audit, power and hash scripts are re-run (§17), and the
committed values are those outputs.

**W_clean rule kept while preparing this text.** No W_clean outcome was read: no return, label, IC, Sharpe, ASR, AUC,
learned coefficient or unit selection that uses one. The inputs were Layer-A tables, the membership table, the
session calendar, the spend ledger and the configuration files. The Layer-A counts quoted here (C2 frames, the
implementations' own C2 eligibility) are judgments, not outcomes.

---

## 0. Summary

### 0.1 What changes against design_v1 §13 and the delta

The user's budget instruction of 2026-09-28 caps total paid LLM spend at $250 (ledger hard cap $245). The ledger
stood at **$185.1424** in 601,615 rows on 2026-09-28 at 08:26:31 UTC (budget `ledger`). That leaves **$64.86** of
the budget and **$59.86** below the hard cap. The confirmatory family is unchanged: **H1–H3 (code C1–C3), Holm over
3, one-sided**. What changes is sample sizes, readers and runs.

| Item | Design | This pre-registration |
|---|---|---|
| H1 units | 400 promotion units × A1–A7 = 2,800 pairs | **200 units × A1–A7 = 1,400 pairs** (MDE 0.050 at π_d = .2, 0.071 at .4; design 0.035–0.050) |
| H1 holistic reader's clean run | H on W_clean × U100 | **H on the in-band set E_t (R^T ≤ K + J = 30) of every decision** (P-1): 3,450 requests, projected $1.36 against the H sub-cap of $1.50 |
| H2 | emu on every C2-eligible item | unchanged: **27,556 units in 115 weeks** (median 232 per week; frame); MDE 0.019–0.030 |
| H3 | ΔIC primary vs V-codeonly over 115 weeks | unchanged contrast, **113 learned weeks**; MDE 0.021 (σ = 0.12, ρ = .8) to 0.043 (σ = 0.242, ρ = .8) |
| Baselines | TAL and MCOT on S50 × 115 weeks; VERB 6,000 items; NG-C2 4,000 items | **TAL S30 × last 60 weeks**, **MCOT S30 × the same 60 weeks** (P-2), **VERB 2,000 items × all five N1 requests**, **NG 500 items**: all generated |
| Cut | — | S2, XWEB paid, HS, VERB-C1I, MCOT-RT, LAP, paid RT-A (moved to NG), PRE's W_clean unblinded arm (D3 Δ_clean), NODEF clean S50, the paid per-reader bound-check sample; RT-S stage 3 (TA-lite under attack, not built); FAITH T4-TAL and T4-H (do not fit, §9) |

### 0.2 Decisions P-1..P-8 (closed; decisions.md 2026-09-28, orchestrator)

| # | Decision | Closed as | Where it is registered |
|---|---|---|---|
| **P-1** | H1's holistic arm: H's clean run on E_t only, its slope learned on in-band names (a coverage restriction, not a code change) | **accepted** | §3.1 |
| **P-2** | The $75 reserve of the pre-unlock list | **dropped; the per-run sub-caps bind**. MCOT sub-cap raised 2.50 → 4.50 **before this commit** (so the single-LLM CoT baseline keeps S30 × 60). No cap is raised from the commit on (§13) | §6, §13 |
| **P-3** | Whether PROBE (D1 + D2) runs | **run** ($0.1598) and completed by a re-ask of 60 scheduled D1 probes ($0.0061, §7.2) | §7.2, §8 |
| **P-4** | RT-A replacement on NG: fixed operator pool, 30 items × ≤ 64 queries | **accepted, implemented and hashed** (spec `e5a1d859…`, pool `067f2eeb…`) | §1, §10 |
| **P-5** | TOST margins and the D1 sampling seeds | **accepted** as proposed | §4.3, §7.2 |
| **P-6** | Prompt-cache routing for `llm-verbalized` | **routing off** (`configs/backends.yaml`; routing only, not in any request key) | §1 |
| **P-7** | Post-freeze run tags in `bank_v1.FROZEN_RUN_TAGS` | **added** (every run tag of this study now refuses a bank other than v1.0) | §1 |
| **P-8** | Every analysis, table and figure entry point written, tested on W_pre or synthetic data, and hashed before the commit | **done** (§14.2, §14.4); the pre-commit audit's findings on that code are fixed (§0.3) | §14 |

### 0.3 Pre-commit audit of 2026-09-28: findings fixed and registered

| Finding | Fix registered here |
|---|---|
| H1 bootstrap clustered on the measurement week (A7's row in another cluster than its unit's six rows) | clusters = the unit's **attack week** t, in H1, every CI of H1, the frontier's ASR and `redteam.run.asr_summary` (§3.1, §14.4) |
| D1 percentile bootstrap over ~6 week clusters too narrow (15.3% false moves) | the rule uses a **two-sided name-permutation test** of AUC = 0.5 at α = .05 (§7.2); simulated false-move rates stated |
| PROBE off the registered D1 schedule (40 scheduled probes unanswered) | the 40 (and the 20 of the W_pre months) **re-asked** exactly as scheduled; the rule reads both generations (§7.2) |
| H2 could not read the VERB generation | VERB reader added and registered as the default (`verb_llm-verbalized_wclean_items2000`); parsed table hashed (§3.2) |
| Result sentences with unchecked clauses | the registered wordings are the bare sentences of §3.1–§3.3; each extra clause is stated only when its own output holds (§4.2); BOUND enters the claim register through `bound_claim.py` |
| H3's drop-13 counted from the first learned week | drop-13 = weeks ≥ **2024-10-04**, a date from the configs, for H2 and H3, a sub-series (§7.1) |
| D1 "exhausted" start undefined on the truncated calendar | the rule's calendar runs through the end of W_clean; exhausted start = **2025-07-03** (§7.2) |
| C1-secondary Holm over the contrasts that ran | Holm over **all 7**; a contrast not run enters with p = 1 (§3.1) |
| Table 2's adaptive and localization columns unreachable | emu adaptive column removed (paid RT-A is cut); RT-A-NG reported item-level in Table 2's note; localization produced by `localization.py`; every input through `check_inputs` (§4.1) |
| DSR trial set not pinned | M = 73 registered trials, latest row per name, ρ̄ over the same set, both raw-M and N̂ DSR reported (§4.4) |
| Sampling text vs code | exact hash strings and the remainder tie-break quoted (§3.1, §10) |
| Output provenance without the import closure | every analysis output records its entry point's closure SHA-256; the W_pre development run is repeated after the code freeze (§14.2) |
| Fig. 3 parsed W_clean cards before the lock check | the lock is checked on the run's manifest window and the file's dates before any card is parsed |
| Table 1 averaged the pre-trade violations | a count |
| Caps quoted against the old MCOT sub-cap | sub-caps re-quoted from `configs/backends.yaml` (§13) |

---

## 1. Frozen configuration

All values are checked in the audit (`provenance` blocks), in `results/freeze/FREEZE.json`, and in the red-team
hashes.

| Item | Identity |
|---|---|
| Question bank | v1.0; canonical SHA-256 `d4f4e44cb7e03d4f1a6d1cb6e3e86cab2627eb302dd81ceee0e27bbe81012ccb`; content = rc7. `FROZEN_RUN_TAGS` (P-7) = MAIN, PRE, TAL, MCOT, VERB, PROBE, FID, FIDX, MEM, H, HS, HSPOT, HGUARD, NODEF, RT-S, RT-A, FAITH, LIVE, CF, S2, XWEB, LAP, VERB-C1I, MCOT-RT, NG: a run with any of these tags refuses a bank other than v1.0 |
| Interface emulator | `llm-logprob` / `emu-v1`, pinned snapshot `gpt-4.1-mini-2025-04-14`, `cfg_hash` `eee07ff63b15a90b`. Prompt-cache routing off (rule O-5); the `llm-logprob` block equals FREEZE.json |
| Verbalized backend (VERB) | `llm-verbalized` / `verb-v1`, the same pinned snapshot, `cfg_hash` `3439578683939242`; prompt-cache routing **off** (P-6). The provider's automatic prefix caching still applies (it does not change answers) |
| Blinder | `blind-text-v4`, `pit-members-v1`, `reading-week-v1`; aliases identity `69a78b8f47b26127` (as freeze_report §3) |
| Data freeze | `archive-a377a4d6abd3bfe3` (the only value in every generation of §14.1) |
| Source tree since the freeze | freeze `freeze-v1.0-src-745fe6660c59`. Changed since (code `src_since_freeze`): `s1/backends.py`, `s1/budget.py` (the ledger's hard-cap default 950 → 245 only), `programs/bank_v1.py` (the `FROZEN_RUN_TAGS` line, P-7), `redteam/run.py` (the sampler's lock guard; the attack-week cluster, §3.1), `live/*` (live readiness, deployment check only). Added: `redteam/{adaptive_ng, faith, projection, stages}.py`, `live/budget.py`. **Nothing under `pipeline/`, `learn/`, `fusion/`, `decide/` or `eval/` changed: the Layer-B decision code is the frozen one** |
| Configs | `configs/system_v1.yaml` and `configs/learning_v1.yaml` unchanged since the freeze (K = 20, J = 10, 15 bps per side, caps 0.10 per name and 0.30 per industry, turnover 0.40, `sizing: reference` (BG-2), `exclusion_rule: anchored` (BG-3), prior sd 0.04, half-life 52 weeks, κ_t = 2·max(s^T_t, 0.01), the V-UAW signs). `configs/backends.yaml` differs from the freeze in the hard cap, the sub-caps and the `llm-verbalized` routing line only. Group SHA-256 of the configs in code `configs` |
| Parser | `gen-parse-v1`, SHA-256 `45a3fe949f47ea4d…` (TAL and MCOT requests, MANIFEST) |
| Attack templates | `rt-templates-v1`, `templates_sha256` = **`2e2a9d4cf9c75fe2097a402f64d54408232743b3dadbd56bf6fd6d7cf9c092c9`**. It covers wording, family specs, salts, obfuscation tables, `attacks.py`, `inject.py` and `display_names_v1.json` |
| RT-A-NG specification (P-4) | `NGSpec().sha256()` = **`e5a1d85998d3907a2b4885e85149f61dd8f520a841c38dbf0480dc91e36458b6`** (spec, objective, rendered pool, templates hash, sources of `adaptive_ng.py` and `adaptive.py`) |
| RT-A-NG operator pool | sha256 of `adaptive_ng.pool_description(NGSpec())` = **`067f2eeb69f18073edc273e9ffab3563c716d2209614db59cc53c9a1db2bd35e`**: 27 frozen payloads × 16 presentation operators × 4 positions (red-team `rta_ng_pool_size`) |
| FAITH | `faith_sha256()` = **`0b64b1ae6866fadd682cc4e7b5b67b69367899cb8dc1da381de31618d92a48fc`** (module source plus the frozen arms, sizes, seeds, mirror map, bin complements and neutral report) |
| Adaptive spec (RB-9) | `AdaptiveSpec().sha256` = `bf7009f2aecf7989e4a4cf4bff24be491b684c96e4001127d566d4c62667739a`, for reference only: the paid RT-A is cut (§10) |

---

## 2. Primary cell (design_v1 §13, unchanged)

- **Protocol P-clean.** Cold start at the first decision of the confirmatory window. W_clean is 2024-07-05..2026-09-11:
  115 decisions, 113 of them learned. The first learned decision is 2024-07-19. The start moves only by the D1 rule
  of §7.2.
- **Universe.** U100, the point-in-time top-100.
- **Backend.** `llm-logprob/emu-v1`, blinded. Generation: `results/judgments/main_emu_wclean_u100`.
- **Allocator.** As design_v1 §6: trusted and untrusted learners, text cap κ_t, K = 20, J = 10.
- **Books.** Signal book D-sel-EW. System book D-sized with g ≡ 1, reference-anchored sizing (BG-2) and anchor-gated
  exclusion (BG-3).
- The primary run is replayed after the unlock with `experiments/run_layer_b.py --bound-check`, so the run directory
  carries the BG-5 check (§3.1 "Bound").
- Everything else is a registered sensitivity or variant (§5).

---

## 3. Confirmatory hypotheses (Holm over H1–H3; family-wise α = .05; one-sided in the stated direction)

The presentation order is H1 → H2 → H3, after the bound. The paper uses the labels H1–H3; code keeps C1–C3.

### 3.1 H1: per-item guarded decomposition vs holistic typed reading, under the same band and cap

**Claim.** ASR_typed < ASR_holistic at the tested attacker and budget. This is not "robust" and not "typed vs
untyped" (FR-10 d, u; RB-18).

**Units.**
- **Size.** **200 promotion stock-weeks** from the learned W_clean weeks. Eligibility (`redteam.run.eligible_units`):
  - learned (not cold) and R^T ∈ (K, K + J] (entry is possible);
  - the name is outside both the typed and the holistic clean D-sel-EW books, checked by replay before any attack.
- **Sampling.** `redteam.run.sample_units(n_promotion=200, n_bound=0, n_demotion=100, n_anchor=25)`, seed
  **`RT-S-www27`**. For each kind the seed is salted: promotion uses `RT-S-www27|promotion`, and demotion uses
  `RT-S-www27|demotion|<stratum>` per stratum (§10). Allocation over the decision weeks t is proportional to the
  number of eligible stock-weeks per week, by largest remainder. Ties among equal remainders go to the week with the
  smaller sha256("<salted seed>|group|<t as 'YYYY-MM-DD 00:00:00'>"). Inside a week, the units are the rows with the
  smallest sha256("<salted seed>|<t as 'YYYY-MM-DD'>|<symbol>"), e.g. sha256("RT-S-www27|promotion|2024-07-19|AAPL").
  Unit ids are "P<YYYYMMDD>-<symbol>". There is no RNG state and no selection freedom after this registration.
- **Timing.** Eligibility needs the learned trusted rank R^T, so units are drawn only after the commit. The sampler
  refuses before the unlock (`run.check_unit_lock`).

**Attacks.**
- One per unit and family, **A1–A7**, in the static setting: 1,400 paired outcomes.
- Templates hashed as §1.
- A7 plants at t, t+1 and t+2. It counts only units that are still eligible at t+2, where success is measured.
  Persistence at t+3..t+4 is S3, not H1.

**Readers.**
- **R-typed:** the full typed pipeline (D0–D6).
- **R-holistic:** H (`H.outlook` over the stock-week's D0-sanitized, blinded items), dropped into the untrusted slot in
  place of the nine news features (`V-reader-H`); sentinel and policy features stay.
- **Deviation P-1 (clean H run):**
  - H's clean answers are generated only for the in-band set E_t (R^T ≤ 30) of every decision: 30 × 115 = 3,450
    requests, tag `H`, projected **$1.36** against the **$1.50** sub-cap (power `cost_projection`);
  - `h_outlook` is `covered = False` for the other names, so H's slope γ̂ is learned from in-band names only;
  - D-sel-EW selects only from E_t (Proposition P1), so H's clean and attacked books are fully determined.
- Both readers use the same κ_t (trusted, reader-independent) and each its own learned slope. This is "under the same
  band and cap".

**Outcome and test.**
- success_r = 1[name ∈ D-sel-EW_r(attacked)], on units whose name is not in D-sel-EW_r(clean).
- The attacker knows the week's learned coefficients (replay; `eval.bound.attacked_week_logits`).
- **Test:** the paired difference ASR_typed − ASR_holistic, pooled over families, on the (unit, family) pairs eligible
  for both readers (`experiments/analysis/h1_asr.py`). The one-sided p comes from the cluster bootstrap of
  `eval.metrics.asr_cluster_bootstrap` with B = 10,000 and seed 0: clusters resampled with replacement,
  p = mean(boot − est ≤ est) (the null-centred bootstrap distribution), 95% percentile CI.
- **Cluster = the unit's attack week t** (`t_attack`; `redteam.run.cluster_weeks`), which nests all seven family
  rows of a unit. A7 is measured at t + 2, so clustering on the measurement week would put a unit's A7 row in another
  cluster than its six other rows. The same cluster is used for every CI of H1 (per family, pooled, harm, rank
  displacement), for the frontier's ASR and for `redteam.run.asr_summary`.
- **Calibration by simulation** (simulation `h1`: 200 units over 113 weeks × A1–A7, unit-level typed − holistic
  propensity d ~ N(0, 1.5), B = 500): the rejection rate at nominal .05 is **0.051** with the attack-week cluster
  (0.106 at .10; 4,000 replications) against 0.067 with the measurement-week cluster; without a unit effect, 0.0585
  (2,000 replications). The test is stated as calibrated to about ±0.01 at .05.

**Power** (power `H1`; plan_v1 formula z_holm·√(2π_d/n), week-cluster design effect 2).

| Pairs | π_d = .1 | .2 | .3 | .4 |
|---|---|---|---|---|
| design 2,800 | 0.025 | 0.036 | 0.044 | 0.050 |
| **registered 1,400** | **0.036** | **0.050** | **0.062** | **0.071** |
| 1,200 (A7 contributes nothing) | 0.038 | 0.054 | 0.066 | 0.077 |

plan_v1.json quotes the design row as 0.035–0.050 at π_d = .2–.4.

**Registered result sentences** (`experiments/report/make_numbers.py`, hashed; macros filled from the registered
outputs only):
- stated (Holm-adjusted p < .05): "Per-item guarded decomposition lowers portfolio-level attack success from
  [ASR holistic]% to [ASR typed]% relative to the holistic reader under the same band and cap (Holm-adjusted
  p = [p])."
- **fallback:** "At the tested attacker and budget, per-item guarded decomposition and the holistic reader did not
  differ detectably (MDE ≈ 0.050–0.071)." Two clauses may be inserted before its final period, each only when its own
  output holds (§4.2):
  - "; the influence bound capped the damage for every reader": only if `confirmatory.json` `claims.BOUND` is True
    (bound.json: BG-5 finds 0 violations and the P0–P5 property tests pass);
  - ", and only per-item readers localized the planted item": only if `localization.json`
    `claim.per_item_readers_localize` is True (the typed reader's localization rate CI lies above 0).

**Secondary** (Holm within the C1-secondary family, **m = 7**; z = 3.29; MDE on the same 1,400 pairs 0.056 at
π_d = .2 and 0.079 at .4). Each contrast is typed vs one of: H-guard; H-spot; nodef; R-med (a $0 replay); R-vote
(a $0 replay); LM dictionary (B10, CPU); TA-lite (100 of the TAL S30 × last-60 stock-weeks × A1, A2, A3, A6 = 400
pairs, MDE 0.104 at π_d = .2). **Holm runs over all seven; a contrast that did not run (or has no eligible pair)
enters with p = 1** and is listed with its status; m_run is reported alongside. **TA-lite under attack (RT-S stage 3)
is not built** (§10), so that contrast enters with p = 1.

The clean runs of H-guard and H-spot are also on E_t (projected $1.00 and $1.72 against sub-caps of $2.00 each).
nodef's clean reference covers only the 200 units' week-t stock-weeks (projected $1.66 against a $3.00 sub-cap).
Scoring for nodef (stated deviation): it has no clean U100 run, so its features are scored with the **typed
reader's** learned slopes and scales, competitors keep their typed clean features, and nodef therefore measures what
the defenses do on the attacked name under the typed allocator (`redteam.stages.score_nodef_batch`).

Also reported with CIs, outside the family: per-family ASR (95% CI half-width 0.088 at π_d = .2, n = 200),
item-level ASR, stock-level Δ combined rank, signed harm and the AgentDojo triad (RB-4), localization
(`localization.py`, §4.1), clean-item false-quarantine and false-removal rates, and the RB-16 checklist.

**Bound (not a hypothesis; must hold).**
- The BG-5 worst-case-reader check runs over every W_clean week and name ($0) in the primary run
  (`run_layer_b.py --bound-check` → `bound_check.json`). Pass criterion: **0 violations** (entry and protection) and
  a trusted side of T-channel features only.
- The property tests of Propositions P0–P5 (`tests/test_band_guard_bound.py`, `tests/test_bound_history.py`) are run
  by `experiments/analysis/bound_claim.py` at the time of the registered analysis; their counts and file hashes are
  written with the claim.
- The paid per-reader bound-check sample (100 × A1–A7) is cut. BG-5 covers the same property for any reader.
- Tightness witnesses per BG-6.

### 3.2 H2: concept validity of typed item judgments (graded by later market reactions)

**Items.** Every MAIN N1 item in W_clean × U100 that meets all of:
- P(about_focal) ≥ .5 and P(new_development) ≥ .5;
- not quarantined and attributes answered (`status.kept`);
- top `event_type` ≠ `stock_activity_or_opinion`, where the top is the argmax and ties go to the first label in bank
  order.

There is one row per (item, focal). The frame has **27,556 units in 115 weeks**: minimum 60, median 232 and maximum
410 per week (frame; power `H2.main_c2_frame`; Layer-A answers only).

**Label.** AR[0,+1] = Σ over d₀ and d₀+1 of (r_i − β̂_i r_SPY), from `learn.labels.abnormal_returns`:
- d₀ is the session of `created_at` if it is before 16:00 ET, else the next session;
- β̂ is the trailing-252-session beta (point in time);
- the label is for evaluation only.

**Statistic and test.**
- The weekly Spearman correlation (average ranks) of E[effect] with AR over that week's units, averaged over the 115
  weeks. A week needs at least 4 finite pairs; a week with a constant side is undefined and counted.
- One-sided Newey–West(4) t.
- Units sharing a (focal, d₀) share their label and stay separate rows, as designed.

**Power** (power `H2`). The per-week null variance is 1/(n_t − 1); τ is the between-week sd.

| | τ = 0 | τ = 0.05 |
|---|---|---|
| units (27,556) | **0.019** | **0.024** |
| distinct (focal, d₀) (13,287; median 116 per week; conservative) | 0.027 | 0.030 |
| design (≈ 260 items per week) | 0.015 | 0.021 |
| W_release (73 weeks), units / (focal, d₀) | 0.023 / 0.033 | 0.029 / 0.037 |

**Controls** (reported): placebo AR[−5,−1]; items created before 09:30 ET only; the prior-look stock-weeks excluded
(§11).

**Sensitivities** (reported): W_release; drop-13 = the frame's weeks ≥ 2024-10-04 (§7.1).

**Secondary, outside Holm.** Resolution-first 8-K agreement (AUC with a week-block bootstrap 95% CI, B = 2,000, seed 0;
Brier; ECE) per implementation: `earnings_results` vs Item 2.02; `deal_or_investment` vs Items 1.01/2.01;
`management_or_labor` vs Item 5.02, each on an 8-K of the focal firm accepted in [created − 2 days, created + 1 day];
and `vs_expectations` (units whose top answer is above or below; score P(above)/(P(above) + P(below))) vs 1[AR > 0].

Agent-level validity (VC-14): sentinel h vs next-week T3 tails; F-B reading accuracy; T answers vs their hand-rule
twins; policy s^pol in high-intensity weeks.

**Per implementation** (VC-13, IF-4; descriptive only):
- **emu** on the full set is the confirmatory row. The same statistic is also computed on the NG units and the VERB
  items.
- **NG** (`typed-local`, `flock-io/this-that-model-1.0@3d927195…`): `ng_typed-local_wclean_items500`, **500 units, all
  finished** (0 abstentions), sample SHA-256 `0c6c3da275ca629d…`, seed `NG-C2-www27`; strata year × emu top event
  type, Hamilton allocation, smallest sha256 rank. NG's own C2 eligibility: 386 of 500 (frame, Layer-A). Statistic:
  the pooled Spearman with a week-block bootstrap (about 4.5 units per week make weekly means unestimable); MDE 0.126,
  95% CI half-width 0.088 (Fisher z); emu-vs-NG agreement: Spearman of E[effect], and κ of top event types
  (agreement-share CI ±0.044).
- **VERB** (`llm-verbalized` / `verb-v1`): `verb_llm-verbalized_wclean_items2000`, **2,000 items** = the NG 500 units
  plus 1,500 more drawn by the same rule (Hamilton allocation of 2,000 minus the units already drawn per cell, floored
  at 0, smallest rank first), sample SHA-256 `708e34519fb9957a…`, **all five N1 requests per item** (about_focal,
  instructs_reader, promotional, new_development, the 13-question attributes request), stage 2 asked for every item.
  VERB's own C2 eligibility: 1,729 of 2,000 (frame). Pooled Spearman MDE 0.063; weekly-mean MDE 0.069; AUC MDE
  0.060 / 0.040 at prevalence .1 / .3.
- **Reading an implementation** (`h2_validity.implementation_items`): NG from its `items.parquet`; VERB from
  `answers.parquet` + `jobs.parquet` + `sample.parquet` (`verb_items`), each request contributing only the questions
  it is used for. An implementation's own C2 eligibility is the production triage (`programs.news.triage`) on its
  answers with emu's N0 decision (both read the N1 state MAIN built from emu's N0), and its top event type ≠
  `stock_activity_or_opinion`. The parsed tables are hashed (frame `implementations`: NG `581deeaeaec37ae1…`, VERB
  `73ef10ad9eb5bea6…`). The registered call reads both by default.

**Registered result sentences:**
- stated: "Typed item judgments track announcement-window abnormal returns (weekly Spearman [ρ], Holm-adjusted
  p = [p])." The clause " and agree with SEC item codes on each implementation" is inserted before the period only if
  `h2.json` `sec_claim` holds: every registered implementation (emu, NG, VERB) ran and every 8-K concept's AUC CI lies
  above 0.5 on its units.
- **fallback:** "Validity was not detected at MDE ≈ 0.019–0.030; SEC concept agreement is reported per
  implementation."

### 3.3 H3: S1-incremental cross-sectional signal

**Contrast.**
- ΔIC_t = IC_t(P_primary) − IC_t(P_V-codeonly).
- Weekly Spearman with r_{i,t} − m_t over U_t (weeks with ≥ 5 names, as Layer B), both on P-clean, same architecture
  and cap.
- Over the **113 learned weeks**.
- One-sided NW(4) t. The Δ log loss (Diebold–Mariano style, P clipped at 1e-4) is reported alongside.

**Power** (power `H3`).

| | ρ = .5 | .8 | .9 | .95 |
|---|---|---|---|---|
| σ_IC = 0.12 (design planning value) | 0.034 | **0.021** | 0.015 | 0.011 |
| σ_IC = 0.242 (W_pre-slice measurement, PRE) | 0.068 | **0.043** | 0.030 | 0.021 |
| W_release (73 weeks), σ = 0.12 / 0.242 | 0.042 / 0.084 | 0.026 / 0.053 | 0.019 / 0.038 | 0.013 / 0.027 |

- ρ on U100 is unknown before the unlock.
- PRE's 0.989 on S30 is inflated: 70 of 100 names have no S1 judgment there (pre_report R-2).

**Sensitivities:** W_release; drop-13 = learned weeks ≥ 2024-10-04 (§7.1); the prior-look stock-weeks excluded.

**Registered result sentences:** stated: "Typed judgments add [ΔIC] rank-IC over an otherwise identical code-only
system (Holm-adjusted p = [p])." **Fallback:** "No increment was detected at MDE ≈ 0.021–0.043 IC."

### 3.4 Multiplicity and reporting rules

- Holm over {H1, H2, H3} at family-wise α = .05 (`experiments/analysis/confirmatory.py`). Each test is one-sided in its
  stated direction. A hypothesis whose analysis did not produce a p enters with p = 1 (m stays 3). The adjusted p and
  the unadjusted estimate with its 95% CI are reported for each.
- Only prediction-level weekly series and the H1 cluster bootstrap carry p-values. There are no stock-week-level
  p-values (design_v1 §13.6).
- H1 counts successes on eligible rows only. Rows excluded for a reader (its clean book already holds the name) are
  reported with their count.
- The result sentences of §3.1–§3.3 are generated by `make_numbers.py` from the registered outputs only (a file of
  the registered window written with the lock open); no sentence is typed by hand.

---

## 4. Descriptive endpoints, claim register, parity margins, snooping budget

### 4.1 PD0–PD3 and the Table 2 columns (FR-3; estimated with CIs, never in the Holm family)

| Id | Status | Estimand (registered) | Data |
|---|---|---|---|
| **PD0** | primary descriptive | Operating profile per question program and reader. Per stock-week: requests, input tokens per request, cached share, $, p50/p95 latency, degraded or invalid share. System rows: MAIN wall-clock; $ per weekly rebalance; Layer-B replay determinism (OPS-3); scaling on S30/S50/U100 subsets of MAIN's requests and by news-volume decile (OPS-2); deployment check (OPS-4; pre-trade violations are a **count**). Units: per stock-week and per weekly rebalance only | `requests.parquet` of every generation; the ledger is the spend of record (MAIN: $119.06 in the ledger vs $109.70 of paid rows; the $9.36 difference is paid work of interrupted units redone free from the cache: audit `spend.difference_by_program`, main_report §4.3) |
| **PD1** | primary descriptive | The compact portfolio table, ≤ 8 rows: D-sized, D-sel-EW, EW-U100, SPY, vol-managed EW-U100, momentum 12-1, LightGBM (B9, P-clean), V-codeonly. Columns: excess return, vol, Sharpe with a studentized circular block-bootstrap CI (block 5, B = 4,999, seed 0; PE-7), max DD, turnover, break-even cost, own-factor selection alpha (PE-3). Caption: DSR (§4.4) and the Random-20 percentile (seeds 0..999). Portfolio IR MDE ≈ **1.90** over 113 learned weeks (power `descriptive`), so this is estimation only | Layer-B replays |
| **PD2** | primary descriptive | The learned pool vs V-EW-log, V-best and V-oracle: IC, log loss, oracle-gap share; leave-one-program-out with refit | $0 replays |
| **PD3** | secondary descriptive | ΔIC, Δ log loss and ΔIR of the full system vs the all-stores-off composite, **restricted to S30** (MEM ran reduced, §9) | MEM + replays |
| Table 2 | supporting | per reader in the same slot, band and cap: clean selection return; static in-band ASR (attack-week-clustered CI); max \|ΔL̄^W\|/2κ; signed harm; **localization** (share of successful attacks whose attacked N23 card ranks a planted item top-1 with positive weight, `redteam.faith.card_ranking`; `experiments/analysis/localization.py`; "n/a" for readers without a per-item channel); invalid share, requests, $ and p95 per stock-week. There is **no adaptive-ASR column for the emu readers** (paid RT-A is cut); **RT-A-NG** is reported in the table's note, item level only (`experiments/analysis/rta_ng_summary.py`: survival of N0 and the guards and the mean E[effect] shift of survivors at 1, 4, 16, 64 queries) | RT-S, RT-A-NG, PD0 |

**Direct memory evidence leads** (MEM-9): store tests, A7 persistence under write gate × N2, write-gate accounting on
clean data, N2 repeat detection. A7-para is not in the 1,400 H1 pairs. It runs only if RT-S has headroom after stage 4
(§13).

### 4.2 Claim register and retirement rules (RP-4)

| Claim | Stated only if | Otherwise (retirement) |
|---|---|---|
| H1 | Holm-adjusted one-sided p < .05 | the fallback sentence of §3.1; "lower ASR" does not appear |
| H2 (emu) | Holm-adjusted p < .05 | the fallback of §3.2. NG and VERB rows are always descriptive, "per implementation" |
| H2 SEC clause | `h2.json` `sec_claim.each_implementation_auc_ci_above_0.5` (every implementation ran; every 8-K concept's AUC CI above 0.5) | the clause is dropped |
| H3 | Holm-adjusted p < .05 | the fallback of §3.3 |
| BOUND (P0–P5) | `bound.json` `claim.bound`: BG-5 finds 0 violations **and** the P0–P5 property tests pass; copied to `confirmatory.json` `claims.BOUND` (outside Holm) | any violation retracts the bound claim (the H1 bound clause and every "the bound caps" statement), and the bug is logged as a deviation |
| Localization | `localization.json` `claim.per_item_readers_localize` (typed rate CI above 0) | the H1 localization clause is dropped |
| PRICE | frontier replays V-J*, V-κ*; clean costs of BG-2 (V-size-v1) and BG-3c | reported whatever the sign; the "band gives up x% of the text's value" clause only if the value-of-text CI excludes 0 |
| PD2 "composition adds" | the NW 95% CI of the weekly log-loss difference (V-EW-log minus the learned pool) lies entirely above 0 | "no detectable difference from equal-weight pooling" |
| PD3 | always reported (a secondary row) | — |
| S5 comparisons with TA-lite / MCOT / H | estimation with CIs on identical names and weeks (PE-11) | no ranking claim beyond the CI; no parity claim unless §4.3 holds |
| Any "calibrated", "robust", "secure", "injection-proof" wording | never (FR-10 d, e) | — |

### 4.3 Parity statements: TOST margins (P-5, accepted)

A parity statement ("comparable", "no worse") is allowed only if a TOST (two one-sided NW(4) tests at α = .05) rejects
both sides at the margin fixed here.
- **Clean weekly rank-IC**, typed vs another reader on identical names and weeks: **δ = 0.02**, just under the design's
  H3 MDE (0.021 at σ = 0.12, ρ = .8).
  - TA-lite and MCOT share the same 1,800 stock-weeks (S30 × last 60); their paired-ΔIC MDE is 0.071–0.132 (power
    `baselines_reduced`, TAL row), so a TOST at δ = 0.02 cannot pass there in practice.
  - The paper makes no parity claim for them.
- **No parity claims** on portfolio metrics (IR MDE 1.9), on per-implementation H2 rows, or on the clean cost of the
  defenses (NODEF's clean S50 run is cut).

### 4.4 Data-snooping budget (PE-1, PE-8)

- **Headline-candidate set for the DSR:** {primary D-sized, primary D-sel-EW}.
- **Registered trial set: M = 73** = the primary's 2 books + 29 configuration variants × 2 books + 13 composition
  variants (`pipeline.variants.registered_names`; `pd1_portfolio.registered_trial_names`). Rows of
  `results/trials.jsonl` outside this set are dropped and listed (e.g. the B11 placebo); a name registered more than
  once (a rerun) keeps its **latest** row. On the registered window every name must be present except the declared
  not-run V-reader-HS/sel_ew and V-reader-HS/sized (HS is cut); raw M stays 73 (the two HS series count although not
  run, which only deflates more). The trial Sharpe variance comes from the present rows.
- **N̂** = ρ̄ + (1 − ρ̄)·M (bailey2014deflated Eq. 9, p.14), with ρ̄ the mean pairwise correlation of the same set's
  stored daily series (primary and configuration variants; composition variants store none). **Both the raw-M and the
  N̂ DSR are reported** (macros `dsr`, `dsrNhat`), and the headline-candidate DSR (M = 2).
- **MinBTL** (bailey2014pseudo Eq. 3.2, p.11). Over 113 learned weeks (2.17 years), 71 null trials have an expected
  maximum annualized Sharpe of about **1.98**; 2 trials, about 0.80. About 2.96 independent trials keep that maximum
  at 1 (power `descriptive.minbtl`). No variant can therefore be promoted to a headline.
- **Question-level ICs:** BH at 10%. **Every other exploratory family:** Holm within the family (design_v1 §13.4, as
  cut in §10).

---

## 5. Registered variants and families

**Configuration variants** (`pipeline/variants.py`; $0 replays; both books registered):
- V-codeonly (the H3 contrast);
- V-joint, V-nocap, V-kappa1, V-kappa4, V-J0, V-J5, V-J20, V-Jinf (frontier: J ∈ {0, 5, 10, 20, ∞},
  κ multiplier {1, 2, 4, ∞});
- V-nodecay, V-static (26 weeks), V-outlook;
- V-size-v1 (BG-2's price);
- V-excl-items, V-excl-bylines, V-excl-collapsed (BG-3c and the SAN-5 sensitivity);
- V-reader-{H, H-guard, H-spot, HS, LM, R-med, R-vote}: HS is cut; H, H-guard and H-spot are covered on E_t only
  (P-1);
- V-drop-{technical, fundamentals, macro, news, sentinel, policy} (X1).

**Composition variants** (`fusion/pools.py`; D-sel-EW registered): V-EW-log, V-EW-lin, V-EW-med, V-MV, V-REP-best,
V-REP-vote, V-SUM, V-DISC, V-UAW, V-2L, V-CDB, V-best, V-GBM. V-oracle is evaluation only.
- **MA-2:** majority ties are broken by V-EW-log.
- **MA-3:** reputation decay 0.9, Beta(1, 1) prior.
- **MA-5:** V-UAW signs are the `uaw_signs` block of `configs/learning_v1.yaml`, frozen by that file's hash.

**Sensitivities** (reported, never selected; `learning_v1.yaml` `sensitivity`): prior sd {0.02, 0.08}; half-life
{26, ∞}; Noul clip 2.94; cap multiplier {1, 4, ∞}; costs {0, 5, 30} bps.

**Fig. 2 frontier replays.** The holistic reader's J/μ replays draw figure points only and are **not registered as
trials**; the typed V-J*/V-kappa* trials already are. y = the paired clean selection return of D-sel-EW against the
same reader's J = 0 book (label-based mean r − m over the book, gross, bps per week).

**Exploratory families after the cuts** (Holm within each):

| Family | Status |
|---|---|
| S2 frontier | replays only; NODEF's clean IC and IR row is cut |
| X1 agents | replays |
| X2 memory | MEM, reduced (§9) |
| X3 System-2 | **cut** (S2) |
| X4 fusion | replays (PD2) |
| X5 backend | VERB 2,000 (generated) |
| X6 Web measurement | paid XWEB cut; see §10 for the NG version |
| X7 overlay | replay; P2 inputs audited, §15 |
| X8 horizon | replays |

---

## 6. Baselines and their reduced samples

All baselines go through the same decision layer, engine and costs. A subset baseline is compared only with the typed
system restricted to the same names and weeks (PE-11, VC-6).

| Id | Baseline | Sample | State (2026-09-28) |
|---|---|---|---|
| B1–B7 | SPY, EW-U100, momentum, reversal, low-vol, cash, Random-20; vol-managed EW/SPY (PE-5) | W_clean × U100 | CPU |
| B8, B9 | logistic, LightGBM on 𝒯_code (P-clean; P-full extra row) | U100 | CPU, after the unlock |
| B10 | LM dictionary (and under attack, C1-secondary) | U100 | CPU |
| B11 | coverage placebo (VC-7) | U100 | $0 |
| B12 | headline sentiment HS | — | **cut** (sub-cap 0) |
| B13 | holistic H | **E_t (30 in-band names) × 115 weeks** | after the unlock (P-1) |
| B14 | TA-lite (TA-style pipeline under our PIT protocol; QB-9 prompts; `gen-parse-v1`) | **S30 × last 60 W_clean decisions (2025-07-25..2026-09-11), 1,800 stock-weeks** | **generated**: `tal_ta-lite_wclean-last60_s30`, $6.7254, audit PASS |
| B15 | Mono-CoT (`mono-cot-v1`, `gen-parse-v1`) | **the same 1,800 stock-weeks as TAL: S30 × last 60 W_clean decisions** (P-2) | **generated**: `mcot_mono-cot_wclean-last60_s30`, 1,800 calls, 0 degraded, 0 invalid; **$4.3424** (projected $4.3034) against the $4.50 sub-cap; validation PASS |
| B16 | FinBERT | — | not run (no torch in the main environment); the paper says so |
| B17 | V-codeonly | U100 | $0 |
| X5 | VERB (verb-v1) | 2,000 C2-eligible units × all five N1 requests (§3.2) | **generated**: 10,000 requests (9,975 paid), 34,000 answers, validity 1.000; **$3.7252** of $4.00, launched in two preflighted parts (the attributes request, then the four single-question requests) |
| NG | typed-local | 500 C2-eligible units, N0 + all five N1 requests per item | **complete**: 500 items, 3,000 requests, 0 abstentions, $0 |

**TA-lite, stated before the unlock** (gens_report §2.4, a Layer-A look). Decisions are 97% HOLD, and the native book
(EW over BUY names, ≤ 20) is empty in 36 of 60 weeks.
- The TA-lite comparison is therefore on the **common-layer book and on p's ranking**.
- The native book is reported as it is.

**Reduced precision** (power `baselines_reduced`):
- TAL and MCOT, S30 × 60: mean-IC 95% CI half-width 0.050 / 0.073 (σ = 0.12 / 0.242); paired ΔIC MDE 0.071–0.132; IR
  MDE 2.61.
- Stated deviation, TAL: the macro-and-policy analyst's request carries no focal firm and is sent once per week (a
  response-cache hit for the other names). The same holds for identical F-B or news blocks.

---

## 7. Windows and the D1 window-start rule

### 7.1 Windows

| Window | Decisions | Role |
|---|---|---|
| W_dev | 20 weeks in 2021–22 | operations only (DEV #1, #2) |
| W_pre slice | 2023-04-14..2024-05-31, 60 | Layer-B development, mechanical checks, power inputs, D3 pre-period (PRE; contaminated for the emulator) |
| W_buffer | 2024-06-07..2024-06-28 | excluded |
| **W_clean** | **2024-07-05..2026-09-11, 115 (113 learned)** | confirmatory; locked until the commit |
| W_release | 2025-04-18..2026-09-11, 73 | sensitivity (after the pinned snapshot's release date). TAL and MCOT lie entirely inside it |
| W_fwd | pseudo-forward 2026-09-18 (in MAIN) and 2026-09-25 (not generated; in LIVE's budget); live 10-02, 10-09, 10-16 | deployment check only |

**Sensitivities (registered):**
- W_release;
- **drop-13: the weeks ≥ 2024-10-04**, the 14th weekly decision of W_clean on the session calendar
  (`common.drop_start`: a date taken from `configs/system_v1.yaml` windows.W_clean, not a count of learned weeks). It
  is a **sub-series** of the primary run (the learner still starts cold at the window start; earlier decisions leave
  the statistic), not a cold-started replay. The same date applies to H2 and H3, and it does not move with the D1
  start;
- excluding the prior-look stock-weeks (§11);
- the D1-based start (§7.2).

### 7.2 D1 window-start rule (design_v1 §2.5, adapted to the reduced probe)

**Probe (as run).** `recall_beats_median` (unblinded, no evidence; `RecallProbeProgram` stage D1), state
{company_name, period = (e_t, e_{t+1})}, tag PROBE, `llm-logprob`/emu-v1:
- **Schedule (P-5 seeds, accepted).** For every month of the probe period **2024-01..2025-06**: the 2 weekly decisions
  of the month with the smallest sha256("D1-www27|" + ISO date), and at each of them the 10 names of U_t with the
  smallest sha256("D1-www27" + symbol). A weekly decision is the last session of an ISO week on the **full** session
  calendar (built through the end of W_clean) and belongs to the month of its own date (`d1_window_start.schedule` on
  `rule_calendar`). 360 probes. The same schedule over the W_pre-slice months (2023-04..2023-12; decisions inside the
  slice) is the W_pre run (180 probes).
- **Generations.** `probe_emu_d1d2` ($0.1598 with D2; answers `275502b8…`) cut its calendar at the month edges, so four
  rule months (2024-07, 2024-12, 2025-03, 2025-04) and two W_pre months (2023-07, 2023-10) asked a month-end
  pseudo-date instead of the scheduled decision. The **60 scheduled probes** (2024-07-26, 2024-12-06, 2025-03-28,
  2025-04-17; 2023-07-21, 2023-10-13; 10 names each) were **re-asked exactly as scheduled** in
  `probe_emu_d1_reask` (`experiments/run_gen_probe_reask.py`; request construction checked key-identical to
  `probe_emu_d1d2` on 12 answered probes): 60 paid requests, 0 degraded, **$0.0061** against a $0.0062 projection,
  preflight against the PROBE sub-cap, validation PASS (schedule closed; answers `a01aa9fb…`). The rule reads the
  union of the two generations (`REGISTERED_PROBES`; a probe answered twice with different values stops it); the
  off-schedule answers are ignored and listed. Every scheduled probe has an answer; `--allow-missing` is not used.

**Rule.**
- **Statistic.** For each month m, the AUC of P(beat) against the realized T1 labels (y = 1[r > m_t] of the same
  stock-week, `learn.labels.LabelStore` over U_t) over the pooled **3-month window {m − 1, m, m + 1}** (about 60 probes
  in 6 decision weeks). A window at the edge of the probe period pools the months that exist and is flagged.
- **Test.** Two-sided **name-permutation test** of AUC = 0.5 (`stats.name_permutation_auc`): one permutation of the
  10 name positions (the hash order) applied in every week of the window, B = 10,000, seed 0,
  p = (1 + #{|AUC_b − 0.5| ≥ |AUC − 0.5|}) / (1 + B). A window **differs from 0.5** iff p < **.05**, and **covers 0.5**
  otherwise. A window whose AUC is undefined (one class) covers nothing. The 95% week-block bootstrap interval (B =
  10,000, seed 0) is written for each window as a description only; it decides nothing.
- **Why a permutation test** (audit 2026-09-28). A window has only about 6 week clusters and the same 10 names every
  week; a percentile week-block bootstrap interval is too narrow there. The name permutation keeps each week's cross
  section and each name's persistence and is exact under the null that P(beat) carries no information about the
  labels. **Null false-move rates by simulation** (simulation `d1`: 6 weeks × 10 names, 1,000 replications, B =
  1,000): iid null, 0.044 for the permutation test vs 0.134 for the bootstrap interval; persistent name effects in
  both P(beat) and y, **0.053** vs 0.316.
- **Start.** If the window centred on **2024-07** differs from 0.5, the confirmatory window starts at the first
  decision of the first month m ≥ 2024-07 whose window covers 0.5; otherwise at the first decision of W_clean
  (2024-07-05).
- **Exhausted.** If no month up to 2025-06 covers 0.5, the start is the first weekly decision of 2025-07,
  **2025-07-03**, and the output says the rule was exhausted.
- **Effect of a move.** The primary cell, the H1 unit frame, the H2 item frame and the H3 weeks all move with the
  start (the primary cell, V-codeonly and the variants are replayed cold-started at the new start; `h1`, `h3` and
  `pd1` refuse inputs that start elsewhere). W_release and the drop-13 date are unchanged.
- **D1 result: not computed before the commit.** Every window from 2024-06 on reads W_clean labels, so the rule is
  computed **right after the unlock, as the first step** (§14.5), by `experiments/analysis/d1_window_start.py` with
  its defaults, before any H1–H3 statistic.
- **Weakness.** A null is only a lower bound on recall, and at this size (60 probes per window; pooled-window MDE
  ≈ 0.21, power `baselines_reduced.PROBE`) the rule is a weak screen. The measured false-move rate of the test is
  stated. The paper says so.

---

## 8. Look-ahead diagnostics (S6)

| Id | Registered design | State |
|---|---|---|
| D1 | as §7.2; also on NG (LA-10, $0) | PROBE generated and completed by the re-ask; the rule runs first after the unlock |
| D2 | `reid_company` (true name + 4 same-FF12 adjacent-size decoys, shuffled by sha256(D2-www27\|t\|symbol), + cannot_tell; chance .2) on every state and `reid_year` (y2019..y2026 + cannot_tell; chance 1/8) on every N1 state; 600 states = 2 periods (W_pre slice from PRE/B's weeks, W_clean from MAIN's) × 3 blinding levels × 100 base states (34 N1 + 33 T + 33 F-B, the smallest sha256("D2-www27\|<period>\|<source>\|<t>\|<symbol>[\|<eid>]") among the stock-weeks the generation asked). Levels: `full` = the production state (200 of 200 keys matched); `names` = the focal firm's SEC names and historical tickers masked (T and F-B states have no free text, so `names` = `full`); `none` = identity revealed as PRE's unblinded arm. Analysis `experiments/analysis/d2_reid.py`: accuracy (top answer = true label; cannot_tell counts wrong; true year = the calendar year of t) with Wilson CI and the exact one-sided binomial p against chance, the cannot_tell share and the mean P(true), per (period, level, question), per source, and pooled; MDE 0.112 per cell and 0.046 pooled. Before any D3 claim | generated in `probe_emu_d1d2` (804 requests); rates computed with the other W_clean statistics after the unlock |
| D3 | **Δ_pre only.** On the W_pre slice, S30, pooled L: Δ_pre = +0.0008 (NW t 0.09; block-bootstrap CI [−0.019, 0.017]); Δ log loss −0.00014; x_mac Δ = 0 exactly (the built-in placebo) (pre_report §7). **Δ_clean and the DiD are not estimable**: PRE's S30 W_clean unblinded arm is cut. The unblinded arm is defined as pre_report §1.1 | fixed |
| D4 | the break test (IC^S1_pre − IC^S1_clean) − (IC^B9_pre − IC^B9_clean) on the **S30 names** of both windows | $0, after the unlock |
| LAP | **cut** (sub-cap 0). DEV found it uninformative on the emulator (P(cannot_know) > .99), so it is reported as a lower-bound null | — |
| Residual identity | blinder residuals in 2.1% of TAL stock-weeks (e.g. "Palo Alto" for PANW; ticker-style mentions in machine-written summaries), shared by every reader of H's evidence. Registered as a D2 stratum; products remain the main re-identification channel (freeze_report W-3) | stated |

---

## 9. Other supporting runs (as run or as registered for after the commit)

| Run | Registered design | Sub-cap | State |
|---|---|---|---|
| FAITH (S4) | arms in this order, each with its null re-ask arm (the unedited request re-asked by a second backend instance without the response cache): **T3** chain intervention on 500 MAIN N3 states (`mirror`: every this-week effect level k → 4 − k; `complement`: the key event's `important` bin → its complement; the first 505 states in the sample order gave 500 exact states); **T2** evidence deletion on 250 learned stock-weeks with ≥ 3 items, stratified by the card's action (top-1, top-3, random-1, random-3 by sha256(seed\|t\|symbol\|eid), keep-top-3); **T4-TAL** on 250 TAL stock-weeks; **T4-H** on 250 E_t stock-weeks. T1, T5 and localization cost $0. Definitions frozen with `faith_sha256` | $3.00 | projection (red-team `preflight`): **T3 runs** ($0.99 point / $1.15 high). T2 runs only if its post-unlock dry run of the exact requests fits the remaining headroom (planning $1.98 / $1.80). **T4-TAL ($1.73) and T4-H ($0.33) cannot fit after T3 and are "not run (budget)"** |
| FID (S8, interface conformance) | **V1** joint vs single on **150** stock-weeks (MAIN's U100 stock-weeks with an asked attributes request, the smallest sha256("FID-www27\|t\|symbol"); one N1 item per stock-week by sha256("FID-www27\|t\|symbol\|eid")): T 6 single re-asks per stock-week, N1 the 13 attributes questions alone (the 4 gate singles are production's own requests); **V2** re-ask floor on 100 production requests (34 T, 33 N1 attributes, 33 N3) × 3 repeats with a throwaway `uid` = sha256("FID-V2\|<MAIN key>\|<repeat>")[:12] appended to the state. Order and label-swap diagnostics not run | $2.50 | **generated**: `fid_emu_wclean_v1-150_v2-100x3`, 3,150 requests (2,550 paid), validity 0.9996, **$0.6021**, sample `215c9503…` |
| FIDX (LA-3) | option-content permutation, 300 N1 + 100 P1 requests × up to 8 shifts, truncated to fit | $3.30 | not run at this text's date; Layer-A only, may run within its sub-cap after a dry-run preflight; otherwise "not run" |
| CF (VC-3) | 3 content-free states per question | $0.15 | as FIDX |
| MEM (X2, PD3) | **arm 1, M-epi off** (N3 re-ask with no N2, no M-epi reads or writes, `earlier_events` empty) on S30 × 115 weeks, MAIN's N01 blackboard replayed (not re-asked); **arm 2, M-work off** (T re-ask, "market regime" not available) on S30 × the last **k = 98** weeks (2024-11-01..2026-09-11; k fixed by the preflight's truncation rule before the run). S30 ΔIC MDE 0.017 / 0.024 at ρ = .95 / .9 (σ = 0.12) | $2.50 | **generated**: arm 1 `mem_emu_mepi-off_wclean_s30` (2,201 N3 requests, validity 0.9993, $1.0126; replay check: MAIN's N2/N3 keys reproduced exactly); arm 2 `mem_emu_mwork-off_wclean-last98_s30` (2,940 T requests, $1.3417); **$2.3543** in total |
| LIVE (S9) | pseudo-forward 2026-09-25, then live 10-02, 10-09, 10-16 (config frozen as `live-v1` before decision #1); shadow books EW-U100, SPY, momentum, B8; deployment-check rows only; cash-only, long-only, pre-trade checker | $6.50 | per-decision projection $1.26 (high $1.82) for U100 (precommit_live), printed and enforced before every paid decision; a decision that does not fit the remaining headroom is "not run (budget)" |
| NG extras ($0; CPU budget ≤ 36 CPU-h, IF-1) | NG-D1 (the D1 schedule on NG), NG-FID determinism (200 requests asked twice), X6-NG and RT-A-NG (§10) | — | items500 done; RT-A-NG ≤ 10.6 h (upper bound, no early stop; with X6-NG and NG-D1 about 17 h against 36 h) |

Jev: no key exists (registration paused), so no Jev row is expected. If a key arrives, `make jev-rows` produces
descriptive rows only. They are billed outside the $250, and nothing in H1–H3 changes.

---

## 10. Cut runs and registered replacements (the paper says so where a reviewer would ask)

| Run | Was | Disposition |
|---|---|---|
| S2 (System-2 + V on S50) | P1, $22.7 | cut: X3 and XP-7/XP-8 not reported |
| XWEB (paid) | P1, $5.2 | cut. **X6-NG**: N0 + guard Nouls on 2,000 pre-cutoff items, stratified by year 2019-07..2024-05, on typed-local (about 2.9 h). Reported as an "S1-flagged share" only (FR-10 p) |
| HS | P0, $4.6 | cut: B12, V-reader-HS and HS under attack removed |
| VERB-C1I, MCOT-RT, LAP | new P0 tags | cut (sub-caps 0) |
| RT-A (paid adaptive attack) | P0, $16.9 | cut on emu. **RT-A-NG** (P-4; spec and pool hashes §1): target = the typed item pipeline on NG as production runs it (D0 and blinding of the edited raw item, N0, the frozen bank's N1 plan, `news.triage`; backend `typed-local` without a ledger); objective = RB-9's typed objective (maximize E[N1.effect] in the promotion direction after survival: not quarantined, an added sentence kept by N0, P(instructs_reader) < .5, P(promotional) < .5, kept for events; a non-survivor scores −4); **30 host items** of the C1 promotion units (`select_items`, seed `RTA-NG-www27\|items`), drawn after the unlock; **≤ 64 queries per item** (16 slots × 4 iterations; early stop at P(level 4) ≥ .7); proposer = a fixed pool of 27 frozen payloads × 16 presentation operators × 4 positions (proposal k uses operator class k mod 3 and the next untried element in the item's sha256 order; parents by RB-9's Gibbs allocation; ≤ 4 additions, ≤ 60 words each). No LLM proposer, no portfolio replay (NG has no Layer-B run), no transfer claim to emu. Reported item level at the budgets {1, 4, 16, 64}: survival of N0 and the guards, mean E[effect] and confidence shift of the best candidate, share of items with a surviving candidate one or more levels up (Table 2 note). The W_pre dry run on 2 items is a mechanics check, not a result |
| PRE W_clean S30 unblinded arm | P0 | cut: D3 has Δ_pre only |
| NODEF clean S50 | P0, $17.4 | cut: nodef appears only in C1-secondary (§3.1). The clean cost of the defenses comes from replays only |
| Per-reader paid bound-check sample | P0 | cut: BG-5's $0 worst-case reader covers every week and name |
| RT-S stage 3 (TA-lite under attack, 100 × A1, A2, A3, A6) | P0 | **not built**: the TA-lite C1-secondary contrast is "not run" and enters the secondary Holm with p = 1 (§3.1) |
| Demotion (RB-2, A8-1/A8-2) | P0: 200 held stock-weeks × {A2-neg, A6-neg, A8}; RB-2 anchor stratum ≤ 50 | reduced to 100 held stock-weeks (50 protected with seed `RT-S-www27\|demotion\|protected`, 50 in-band held with `RT-S-www27\|demotion\|in_band`; anchor stratum ≤ 25 with `RT-S-www27\|demotion\|anchor`; the §3.1 hash rule) × {A8-1, A8-2}, stage 5 of RT-S. It runs only if the headroom after stages 1, 2 and 4 covers its preflight |

---

## 11. Prior looks (declared; validity M4)

**At W_clean outcomes** (before this pre-registration):
1. **Pilot** (2026-09-25), `results/pilot/news_signal_pilot.parquet`:
   - 20 large caps × 30 weeks in 2024-07-05..2026-08-21, **600 stock-weeks**;
   - company name withheld in the state but **headlines unblinded**; hindsight name selection;
   - mean rank-IC: logprob 0.048 (t 1.12), verbalized 0.018 (t 0.40) (`results/pilot/README.md`).
2. **Wording smoke test** (2026-09-25): 9 items created 2025-04..05 for AMD, JPM and GILD (3 each; Layer-A answers
   only).
3. **Classical-factor ICs on W_clean** (evaluation_first §7.4): momentum 0.032 (t 1.20), reversal 0.012 (t 0.56),
   low-vol −0.031 (t −0.94).
4. **`results/baselines_classic.json`**: B1–B7 over 2020-01-01..2026-09-18, which includes W_clean.

**Exclusion sensitivity.** Every confirmatory result is also reported without the prior-look stock-weeks in
`results/prereg/prior_look_stock_weeks.csv`, SHA-256 `9c4ebae001b98b80…` (hashes `prior_look_stock_weeks`):
- the 600 pilot keys;
- AMD, JPM and GILD in the 10 decision weeks 2025-04-04..2025-06-06;
- 549 of the 630 rows are in MAIN's frame.

Items 3 and 4 are portfolio-level and cannot be excluded by stock-week. They are declared only.

**At W_clean Layer-A judgments** (allowed; not outcomes):
- MAIN's operating profile and gate firing rates (main_report §5);
- TAL's judgment profile (97% HOLD, BUY in 24 of 60 weeks) and its leak probe;
- NG's non-degeneracy check on 20 items and its sample frame;
- the freeze's one exploratory name-mention count over the whole archive (freeze_report §8 item 2);
- the pre-registration audit and the C2 frame counts; the implementations' own C2-eligibility counts (NG 386 of 500,
  VERB 1,729 of 2,000; frame);
- the paid pre-commit runs' validation summaries (validity, degraded counts, tokens, spend); no D1 AUC, D2 rate,
  FID V1/V2 statistic, H2 row or MEM ΔIC was computed.

**At W_pre outcomes** (development window; informed this text):
- PRE's mechanical checks M1–M7, all PASS;
- M1b: a planted IC-0.10 signal is recovered at only 0.37 of its direct IC by the full frozen learner over 58 weeks.
  This is a property of the learner, stated in the paper;
- the primary cell's W_pre IC of −0.031 (descriptive, contaminated);
- D3 Δ_pre;
- the power input σ_IC = 0.242;
- the analysis entry points' development run on the W_pre slice (`experiments/analysis/run_wpre.py`; pipeline
  checks, including the D1 rule on the real W_pre probe answers centred on 2023-07 and D2's W_pre rates).

**DEV #1 and #2** (W_dev, 2021–22): operational metrics only; no outcome was read.

---

## 12. Accepted side costs of the isolated gates (fixed; reported as measured, never tuned)

These are the decisions.md row of 2026-09-28, with MAIN's W_clean rates, which are Layer-A rates and not outcomes
(main_report §5.3).

| Gate | DEV | MAIN (W_clean) |
|---|---|---|
| N0 quarantine | — | 4.58% of items |
| Instruction guard quarantine | 1.48%, incl. some ordinary analyst notes | 1.97% of asked items |
| Promotional flag | 2.4% | 2.72% of non-quarantined items |
| Sentinel escalation (h ≥ .5) | 10.6% of stock-weeks | 10.34% |
| Stage-2 gate pass | — | 75.7% |
| LAP probe | uninformative (P(cannot_know) > .99) | cut |

**Also fixed:**
- the clean costs of V-size-v1 and of the anchor rule (BG-3c), reported in the Fig. 2 caption (primary minus
  V-size-v1 and primary minus V-excl-items: the weekly excess return of D-sized, compounded over each holding week);
- the clean false-quarantine and false-removal rates, reported next to H1.

---

## 13. Stop and budget rule

1. **Caps at the commit** (`configs/backends.yaml`, re-quoted from the file; budget `runs`). Ledger hard cap
   **$245** (user budget $250). Enabled sub-caps: MAIN 132.00, PRE 31.50, TAL 12.00, RT-S 12.00, LIVE 6.50, **MCOT
   4.50**, VERB 4.00, FIDX 3.30, NODEF 3.00, FAITH 3.00, FID 2.50, MEM 2.50, HSPOT 2.00, HGUARD 2.00, H 1.50, PROBE
   1.50, SMOKE 0.30, CF 0.15. Disabled (0): HS, RT-A, S2, XWEB, VERB-C1I, MCOT-RT, LAP, DEV, DEV0, DEV1, DEV2,
   DEV2OPS.
   - **MCOT's sub-cap was raised 2.50 → 4.50 before this commit** (decision P-2, decisions.md 2026-09-28), so that the
     single-LLM CoT baseline keeps S30 × 60. That is the only raise, and it is recorded here as a pre-commit decision.
   - **From the commit on, no cap is ever raised.** Lowering a cap (e.g. to 0 for a completed run) is allowed. A
     sub-cap of 0 disables the run.
2. **Spend at this text's date** (budget `runs`; ledger $185.1424): MAIN 119.0581, PRE 31.1928, TAL 6.7254, MCOT
   4.3424, VERB 3.7252, MEM 2.3543, FID 0.6021, PROBE 0.1658 (D1 + D2 $0.1598, re-ask $0.0061), SMOKE 0.0054; DEV runs
   and pre-scheme rows $16.97 (historical). H, HSPOT, HGUARD, NODEF, RT-S, FAITH, LIVE, FIDX and CF have spent $0.
3. **Preflight before every paid run.**
   - It prints the projected cost against the run's sub-cap headroom, the hard cap and the remaining total.
   - The projection comes from a free dry run: the exact requests go to a fake client calibrated on measured
     per-request costs from the ledger and the DEV/MAIN reports.
   - The run aborts if the projection exceeds the sub-cap headroom, or if it would take the ledger above $245. The paid
     process is bounded by a guard of that headroom; the ledger's own sub-cap and hard-cap checks apply to every call.
   - **No paid smoke tests.** Code is tested on the mock backend or by cache replay. SMOKE ($0.30; $0.0054 spent) is
     the only paid smoke allowance.
4. **Order after the commit** (projections: power `cost_projection`, precommit_redteam, precommit_live):
   1. LIVE: $1.26 per decision (high $1.82), calendar-bound, against $6.50.
   2. H, H-spot, H-guard clean on E_t: $1.36 / $1.72 / $1.00 against $1.50 / $2.00 / $2.00; each only if its own
      preflight fits, and the list stops at the first abort.
   3. RT-S stages 1 (typed + H), 2 (H-guard + H-spot), 4 (nodef attacked), 5 (demotion), each only if its preflight
      fits the remaining RT-S headroom of $12.00. The W_clean-equivalent projection from the W_pre mock rehearsal is
      $7.33 cumulative (gates open, an upper side); **the binding projection is the post-unlock dry run on W_clean**.
      Stage 3 is not built.
   4. NODEF clean on the unit stock-weeks: $1.66 of $3.00.
   5. FAITH: T3 ($0.99 / $1.15 high), then T2 only if its post-unlock dry run fits.
   6. RT-A-NG ($0; CPU), X6-NG, NG-D1.
   7. FIDX and CF, if the orchestrator runs them (Layer-A; their own preflights).
5. **Worst case.** If every run with a positive sub-cap spent all of its remaining headroom, the ledger would end at
   **$241.22** (budget `worst_case_ledger_if_every_enabled_run_spends_its_headroom_usd`; $56.08 of headroom), inside
   the hard cap. The sub-caps, not a reserve, are the binding limits (P-2).
6. **Stop.** A run that does not fit is not replaced by a different design after the unlock. It is reported "not run
   (budget)" with its registered design. No run starts after 2026-10-12, except LIVE decision #3 (10-16).
7. **Detached processes.** A detached paid run is stopped by PID with its whole process tree, and checked with `ps` and
   the ledger's latest timestamp (decisions.md 2026-09-28 lesson; main_report §6 item 1).

---

## 14. Hashes of what the analysis runs on

### 14.1 Generations (`results/analysis/GENERATION_HASHES.json`; all generations `727ea674a7e2ea93…`)

| Generation | Rows | Group SHA-256 (16) | Answers / requests table SHA-256 (16) |
|---|---|---|---|
| MAIN `main_emu_wclean_u100` (116 weeks × U100) | requests 835,540; answers 1,585,879; evidence 749,052 | `62bf12387c0a3cca` | `72027c5c3d869b71` / `93df749c2101d678` |
| PRE/B `pre_blinded_emu_wpre_s30` | requests 66,479; answers 267,795 | `400e2ba5ffb1c770` | `5ceb4c6ae4364cb3` / `48c2eb5d3f2c9381` |
| PRE/U `pre_unblinded_emu_wpre_s30` | requests 65,070; answers 264,296 | `e9b95ec9bd20580a` | `d4c5c5f03d1fa949` / `3897777f1e7482e7` |
| TAL `tal_ta-lite_wclean-last60_s30` | requests 14,400; answers 1,800 | `8d99d32c1e7cf92c` | `d0ae8686d18308c3` / `83ca159d19e4e7d8` |
| NG `ng_typed-local_wclean_items500` | 500 items; requests 3,000; answers 9,930 | `fc3caa9d9f0fa97a` | `10229daeb95bd7c8` / `ee131aaee50edfd5` |
| MCOT `mcot_mono-cot_wclean-last60_s30` | requests 1,800; answers 1,800 | `c5da94acab78962f` | `42000375a756f3ef` / `29352dba54af903d` |
| VERB `verb_llm-verbalized_wclean_items2000` | requests 10,000; answers 34,000; sample `708e3451…` | `6149417842ecfa1b` | `92048746aead9cb9` / `3d4e17d2243f286a` |
| PROBE `probe_emu_d1d2` | requests 1,344 (D1 540; D2 804) | `3cb95e4785a75770` | `275502b89816c0ba` / `cae47deacff25de4` |
| PROBE re-ask `probe_emu_d1_reask` | requests 60 | `13c8c03471520e58` | `a01aa9fb8dd66dcc` / `6c1e6aada9517952` |
| FID `fid_emu_wclean_v1-150_v2-100x3` | requests 3,150; answers 5,145 | `764bf5994f98fb8a` | `a01a7ae0ee0d71e4` / `2e44929a2b35a8b0` |
| MEM arm 1 `mem_emu_mepi-off_wclean_s30` | requests 2,201 | `912210a37fcee683` | `dfe0ef5ab392fbc4` / `2fda27dd3e86f3d1` |
| MEM arm 2 `mem_emu_mwork-off_wclean-last98_s30` | requests 2,940 | `700433470334cb33` | `7c5632a065371f64` / `7d4d138d07bb47ac` |

The full SHA-256 of every table, manifest and blackboard file is in `results/analysis/GENERATION_HASHES.json`; the
commit re-runs it and quotes the commit-time values.

### 14.2 Analysis and red-team code (`results/analysis/CODE_HASHES.json`, 120 files; values of 2026-09-28)

| Group | SHA-256 |
|---|---|
| Registered: analysis + report entry points and the red-team/FAITH drivers, with their import closures | `3706f705f01a62c7ce665337e4270a60341c55c560dd845b14d5eb92b8fa8934` |
| Analysis and report entry points only (h1, h2, h3, d1, d2, confirmatory, bound_claim, localization, rta_ng_summary, pd0–pd3, frontier, tables 1–4, figs 2–3, make_numbers) | `1aee79aaaa95b0584914aaeef8da2c43d6055ce541cc969c53edf437131586e1` |
| Red-team and FAITH drivers (rts, rta_ng, calibrate_tokens, precommit_hashes, faith; the PROBE re-ask runner) | `9448d4f79fb234e13adc8e3787d8bb3faa2e3851b52cc5419a7fef8ec27810cc` |
| All hashed files, including the development tools (run_wpre, feature_plants, code_hashes, sim_null_rates, budget_snapshot) | `6b5837cdfbb5e3610d41497a008b6ee7241ff6859b6c496c03900cc526f45e6f` |
| Configs (system, learning, backends, pyproject) | `2cccc776e71b6041ff0d875e854f6807101a9c926003346d54cf13b52f94eb70` |
| Red-team group of 15 files (red-team `group_sha256`) | `59bde72e5c74f037515b6ea872e9b6ea17ac72e31be457479383dde03186c4fa` |
| Lock (`pipeline/lock.py`, unchanged since 2026-09-25) | `601685954be906f0…` (hashes `analysis_code`) |

Each entry point has its own `closure_sha256` in CODE_HASHES.json, and every analysis output records the closure
SHA-256 of the entry point that wrote it (`common.write_result`), so an output is tied to the exact code. The W_pre
development run (`run_wpre.py`) was repeated after this code freeze. `code_hashes.py --check` must be clean at the
commit.

### 14.3 Entry points (all written, tested on W_pre or synthetic data, and hashed; P-8)

| Entry point | Prereg | Output |
|---|---|---|
| `experiments/analysis/d1_window_start.py` | §7.2 | `d1.json` (read by every other registered analysis through `common.d1_start`) |
| `experiments/analysis/h1_asr.py` | §3.1 | `h1.json` |
| `experiments/analysis/h2_validity.py` | §3.2 | `h2.json` (`--frame-only`: `h2_frame.json`, Layer A) |
| `experiments/analysis/h3_signal.py` | §3.3 | `h3.json` |
| `experiments/analysis/bound_claim.py` | §3.1 Bound, §4.2 | `bound.json` |
| `experiments/analysis/confirmatory.py` | §3.4, §4.2 | `confirmatory.json` (Holm over H1–H3, claims incl. BOUND) |
| `experiments/analysis/localization.py` | §4.1 | `localization.json` |
| `experiments/analysis/rta_ng_summary.py` | §10 | `rta_ng.json` |
| `experiments/analysis/d2_reid.py` | §8 | `d2.json` |
| `experiments/analysis/pd0_operations.py`, `pd1_portfolio.py`, `pd2_composition.py`, `pd3_stores.py`, `frontier.py` | §4.1, §4.4, Fig. 2 | `pd0..pd3.json`, `frontier.json` |
| `experiments/report/table1..4_*.py`, `fig2_frontier.py`, `fig3_decision_card.py`, `make_numbers.py` | floats and macros | `paper/tables`, `paper/figures`, `paper/generated/numbers.tex` (registered outputs only) |

Not written by these entry points (their macros stay placeholders and their cells show "---" unless an owner adds a
hashed script before the commit): FID V1/V2 summaries (`confJointAgree`, `confReaskFloor`), the OPS-3 replay check
and OPS-4 `deployment.json` fields (PD0 passes them through), share of D_t (`budgetUse*`), anchor rates,
reachable-byline ASR, TA-lite/Mono-CoT clean selection return.

### 14.4 Analysis definitions fixed in code (quoted here; each is what the hashed code does)

1. **H1.** Cluster = the unit's attack week (`t_attack`); B = 10,000, seed 0; percentile CI; p = mean(boot − est ≤
   est). C1-secondary Holm over all 7 registered contrasts, not-run at p = 1.
2. **H2.** A week needs ≥ 4 finite pairs; average ranks; a constant side is undefined and counted. "Pre-open" = an
   item whose `created_at` in ET is before 09:30. The 8-K window is [created − 2 days, created + 1 day] on
   `acceptanceDateTime`. vs_expectations: units whose top answer is above or below; score P(above)/(P(above) +
   P(below)); label 1[AR > 0]. Per-implementation pooled Spearman: week-block CI (B = 2,000, seed 0) and Fisher-z CI.
   8-K AUC CIs: week-block, B = 2,000, seed 0.
3. **H3.** A weekly IC needs ≥ 5 names (Layer B's own rule). DM test on P with clipping 1e-4.
4. **D1.** As §7.2 (name-permutation test, B = 10,000, seed 0, α = .05; descriptive week-block interval B =
   10,000, seed 0; edge windows pool the months that exist and are flagged; exhausted start 2025-07-03).
5. **Drop-13.** Weeks ≥ 2024-10-04 (`common.drop_start`), a sub-series, for H2 and H3.
6. **PD1.** Sharpe CI: studentized circular block bootstrap, block 5, B = 4,999, seed 0; Sharpe with ddof 1.
   VM-EW-U100: c_d over every session since the archive start (2018). Momentum 12-1: the top 20 of U_t by the trusted
   code feature `mom_12_1`. LightGBM: P-clean, cold-started at the window start, on the generation's 14 code
   features. Code-only: V-codeonly's D-sized book. DSR: §4.4. Random-20: seeds 0..999 with `numpy.default_rng`, net
   of 15 bps.
7. **PD2.** "Composition adds" only if the NW 95% CI of the weekly log-loss difference (V-EW-log minus the learned
   pool) lies entirely above 0.
8. **Fig. 2.** §5 (paired clean selection return vs the same reader's J = 0 book; holistic J/μ replays are figure
   points only); clean costs as §12.
9. **Fig. 3 (XP-4).** "In-band" = band `eligible` in a learned week; "kept items" = `evidence.screening.n_kept`; the
   10-02 live decision's in-band name with the largest |L̄^W| and ≥ 1 kept item (ties to the smaller
   sha256(symbol)); else the W_clean card with the smallest sha256("fig3-www27|" + ISO date + "|" + symbol) among
   in-band names with ≥ 3 kept items. The lock is checked before any card is parsed.
10. **Table 2.** The "displacement of the uncapped L^W" column is replaced by max |ΔL̄^W|/2κ (the P4 witness; RT-S
    scores do not carry the uncapped attacked logit). The cut headline-tone row is replaced by the LM dictionary.
    Localization and RT-A-NG as §4.1.
11. **Localization.** A successful, eligible promotion row (offset 0) of a per-item reader (typed, R-med and R-vote
    read the typed arm's cards; nodef its own) is localized iff the attacked N23 card of (eval_t, symbol) ranks a
    planted item (the planted item, or the host an edit family rewrote) top-1 with positive weight
    (`faith.card_ranking`). Rate per reader with an attack-week-clustered CI (B = 10,000, seed 0).

### 14.5 Order after the unlock (every step $0 except the paid runs of §13)

1. `lock.write_unlock` (orchestrator), after §17.
2. **`d1_window_start.py` (first).** If the start moved: replay the primary cell, V-codeonly and the variants
   cold-started at the new start.
3. The primary cell with `run_layer_b.py --bound-check`, V-codeonly and the registered variants → `bound_claim.py`.
4. `h2_validity.py`, `h3_signal.py`.
5. The paid runs of §13 item 4 in order; RT-S (`rts.py et`, `clean-readers`, `sample`, `nodef-clean`, `stages`, `run`,
   `score`) → `h1_asr.py`, `localization.py`; RT-A-NG `run` → `rta_ng_summary.py`.
6. `confirmatory.py`.
7. `d2_reid.py`, `pd0..pd3`, `frontier.py`, the FAITH analysis.
8. The report scripts, then `make_numbers.py`.

---

## 15. Completeness audit of the inputs (summary; details in the audit JSON, `docs/impl/prereg_report.md` and
`docs/impl/precommit_paid.md`)

**Overall result: PASS** for every generation of §14.1.

**MAIN**
- Rows: 116 weeks; 11,600 stock-weeks each with exactly one T and one FB request, one risk row, 14 code features and
  17 features (LM 1, PX 2, N01 2, N23 12).
- Programs: M once, P2 once and P1 at least once per week.
- Validity: 99.9835% (N3 lowest at 99.675%; lowest N3 program-week 98.485%). Every degraded request made 3 attempts.
- Provenance: one bank hash, pinned model on all 361,708 paid rows, `emu-v1`, one data freeze, `cfg_hash` and aliases
  equal to FREEZE.json.
- Duplicates: 191 same-key groups inside a stock-week (N0 77, N1 5, P1 1, P2 108). Each is several distinct items with
  identical text, of which at most one row was paid; none is unexplained.
- P2's P1-history answers: 2,937 keys are answered in PRE's tables. **240 (the four W_buffer weeks) exist only in the
  response cache.** These are X7 overlay inputs; export them before the cache can be lost.

**PRE/B and PRE/U**: 60 weeks × S30; code features and firm info over all of U_t, by design; validity 99.985% /
99.990%; tables equal the ledger to the cent.

**TAL**: 1,800 stock-weeks × 8 roles, 0 degraded calls, 3 invalid analyst stance lines; tables equal the ledger
($6.7254).

**Pre-commit paid generations** (each generation's `validation.json`, precommit_paid): PROBE, MCOT, VERB, FID, MEM
arm 1 and arm 2 and the PROBE re-ask all PASS: complete against their plans; pinned (model, prompt version, frozen
bank) on every row; only their own ledger tags; validity ≥ 98%; ledger ≥ paid rows (equal to the cent). One tag note:
MEM arm 1's ledger rows carry `MEM/MEM/N3` (the run key is still the first segment).

**NG**: 500 of 500 items, 3,000 requests, 0 abstentions, pinned revision, bank frozen, $0; every emu key found in
MAIN's requests.

**Ledger**: $185.1424 in 601,615 rows; every run is within its sub-cap (budget `runs`).

**At the commit** (§17 step 2): `experiments/prereg/audit_generations.py` also audits MCOT, VERB, PROBE, the PROBE
re-ask, FID and MEM arms 1 and 2. It checks plan, answers, validity, provenance and spend, and rebuilds from primary
inputs: the D1 schedule and the union of the two PROBE generations (540 scheduled, 0 missing), D2's base states, MCOT
on TAL's stock-weeks, the VERB and FID samples, and MEM's stock-weeks. All PASS (audit `generations`).

---

## 16. Deviations and change control

- After the commit, any change is a bug fix, logged in `docs/design/prereg_deviations.md` with its reason, date and
  effect.
- The stated deviations of this pre-registration are:
  - P-1, H on E_t;
  - nodef scored with the typed slopes;
  - the D1 3-month pooling and its name-permutation test; the PROBE re-ask of the 60 scheduled probes the first
    generation asked on month-end pseudo-dates;
  - D2's operationalization (§8; the `names` level masks the focal firm only);
  - D3 with Δ_pre only;
  - FAITH, FID and MEM reductions (FID V2's appended `uid`; MEM arm 1's replayed N01; arm 2's k = 98);
  - VERB sent in two launches (the attributes request, then the singles) in one generation;
  - TAL's shared macro request;
  - NG N1 states built from emu's N0 decision (ng_report §6);
  - PRE's unblinded arm mechanics (pre_report §11);
  - RT-S stage 3 not built; FAITH T4 cut; HS cut (the two HS trial series count in M = 73);
  - the pre-commit MCOT sub-cap raise (P-2).
- Editing this file or `configs/prereg_v1.yaml` after the unlock closes the lock (`pipeline/lock.py`) until the
  change is logged and the key re-issued.
- **Commit-time changes (2026-09-28, §17 steps 1–3).** Steps 1 and 2 re-ran every hash, audit, power and budget
  script. No quoted number changed. The text changed in three places:
  - the status line and the §17 heading: the commit is on 2026-09-28, earlier than the 2026-10-02 plan, because all
    inputs were ready;
  - §15: "For the commit" (the audit still listed MCOT, VERB, PROBE, FID and MEM as not run) is replaced by the
    commit-time audit of those generations and the PROBE re-ask, all PASS;
  - this entry.
- **Output fields that moved at the re-run but are not quoted here:**
  - power `cost_projection.NG_cpu`: NG items500 finished at 2026-09-28T07:34Z, so the measured 26.58 s per item
    gives items500 3.69 h (was 3.31 h). The §9 total becomes 17.4 h, still "about 17 h" against 36 h;
  - power `cost_projection.MCOT_reduced.sub_cap` 2.5 → 4.5 (the P-2 raise);
  - audit: NG complete and the ledger at $185.1424;
  - hashes: the `trees`; the `red_team`, `drivers` and `configs` groups, through the pre-commit changes that §0.2,
    §0.3 and §1 already describe (`redteam/run.py`, `make_numbers.py`, `backends.yaml`); and the `prereg_tools` group
    (the audit extension). The quoted `lock` group and the prior-look CSV are unchanged.

## 17. Commit procedure (executed on 2026-09-28, earlier than the 2026-10-02 plan, because all inputs were ready)

1. P-1..P-8 are closed (§0.2). Confirm that no file hashed in §14 changed (`code_hashes.py --check` clean;
   `code_hashes.py --generations` equal to §14.1; `experiments/redteam/precommit_hashes.py` equal to §1).
2. Re-run `audit_generations.py` (with MCOT, VERB, PROBE, the re-ask, FID and MEM audited), `power_prereg.py`,
   `prereg_hashes.py` and `budget_snapshot.py`; update the quoted values if any changed.
3. Write `configs/prereg_v1.yaml`, the object the lock hashes. It holds at least this file's SHA-256, the SHA-256 of
   the audit, power and hash JSONs, of CODE_HASHES.json, GENERATION_HASHES.json and precommit_hashes.json, the attack
   templates hash, the RT-A-NG spec and pool hashes, the FAITH hash and the bank hash. That way an edit of this
   Markdown also invalidates the key.
4. Print both SHA-256 values (this file and `configs/prereg_v1.yaml`; AN-3), commit and push (private repository).
5. Only then run `lock.write_unlock` (orchestrator). Then the order of §14.5: the D1 rule first, then H1–H3.
