# Deviations from the pre-registration v1 (change control after the commit)

Pre-registration: `docs/design/preregistration_v1.md`, SHA-256
`b8bf8cf6e7135b963cd2069f83214019be4df6ba9bebabbfd81dd41470984b9a`. Lock object: `configs/prereg_v1.yaml`, SHA-256
`15dd7a489205b89fa58a5fc135067ba686ca47a248e90047c838ea31c9273dbc`. Unlocked 2026-09-28T09:42:22Z.

Rule (prereg §16): after the commit, every change is a bug fix or an operational deviation. Each is logged here with
its reason, date and effect. Neither file above is edited. This file is outside the lock's hash.

---

## DEV-1 (2026-09-28): the primary cell's Layer-B run waives the relative `coverage` gate for one quiet week

**Where.** Prereg §14.5 step 3: the primary cell on W_clean with `experiments/run_layer_b.py --bound-check`,
V-codeonly and the registered variants (§2, §5).

**What happened.** The registered call
`run_layer_b.py --gen-id main_emu_wclean_u100 --window W_clean --variants all --composition --bound-check --attribution --sharpe-tests`
was refused by the Layer-B input gate (`pipeline.layer_b.input_report`, check `coverage`) before any decision was
computed. Nothing was written and no trial was registered: `results/eval/*/W_clean__*` did not exist and
`results/trials.jsonl` had no W_clean row afterwards.
The flags were:
`f_imp, f_cont, f_rev, f_sur, f_net, f_und, f_story @ 2026-01-02`.

**Why the gate fired (Layer-A judgments only; no outcome read).** Source: `results/judgments/main_emu_wclean_u100/features.parquet`.
- The full-run rule marks a week as a hole when a feature's value coverage is below 0.5 × its median weekly coverage.
- The seven N23 news features have median value coverage 0.61 over the 115 W_clean decisions. The hole threshold is
  therefore 0.305.
- In the decision week of 2026-01-02 (the New Year holiday week), 30 of 100 names had a covered N23 value (0.30).
- The same holiday week a year earlier (2025-01-03) had 0.32, just above the threshold. The next-lowest weeks are
  2024-12-27 (0.36) and 2025-12-26 (0.42).
- **Row coverage is 1.0 for every feature in that week.** Layer A wrote a row, a value or an abstention, for all
  100 names. So the absolute Layer-A week check (`layer_a_weeks`, the crash detector) passes, and no other input
  check failed.
- The week is a quiet news week, not an outage or a crashed shard.

**What was done.** The registered call was re-run with the gate's own waiver, `--allow-missing coverage`. The waiver
is `DecisionConfig.allow_missing`, the designed mechanism: `layer_b.py` records it with every trial, in
`metrics.json` `inputs` and in `inputs_manifest.json`. No code was changed.

**Effect.**
- **Decisions.** None change. `allow_missing` is read only by the input gate (`layer_b.py` lines 614–626 and 750–752)
  and by the trial record (line 1068). It never enters a decision.
- **Configuration hash.** It changes, because `allow_missing` is a field of the hashed dataclass. The primary config
  without the waiver is `2052f09a187d5196b99e2a630e4c69f8e71058f52545ebb4ffe82091faaaa17e`; with the waiver it is
  `4ea23931a045dd49a3f22640ca83c51cf262bba1a812e0f685c76aea990b9273`. Only the field `allow_missing` differs. Every
  variant inherits the waiver. The pre-registration pins no Layer-B configuration hash.
- **Where the effect shows.** Only the run directory's `<cfg16>` and the trial rows' `cfg_sha`. Trial names, and so
  the registered trial set (M = 73, §4.4), are unchanged.

**Code hashes: unchanged.** Before = after:

| File | SHA-256 |
|---|---|
| `experiments/run_layer_b.py` | `e2dcad233ebb9285b8478b2ed5559447063b8cf8076fedf580456a7a97ca2b7c` |
| `src/jevtrader/pipeline/layer_b.py` | `1cd7c461541f9c9076dbee46db77c622c3f8ae1b92ddf88a722cc54b6d69328a` |
| `src/jevtrader/pipeline/variants.py` | `9996f53ad86b8e2b046f6c68074657c4f7d2be8cc7da4b3cfaeccc73106b7ed3` |

`code_hashes.py --check` reported no stale file or config before the run.

**Downstream consequence (open; for the owners of §14.5 steps 5 and 7).** Two hashed drivers build the primary
configuration without a waiver and call `evaluate` on W_clean, so they will meet the same gate refusal:
- `experiments/redteam/rts.py` `layer_b()`, the typed clean log used by `et`, `sample` and the later stages. Its
  `--allow-missing` option reaches the reader runs only (`stages.reader_log`), not the typed run at line 104.
- `experiments/analysis/frontier.py` line 286; it has no waiver option.

Each will need the same waiver passed through. That is a minimal code change to hashed files, so it must be logged
here as its own entry with before and after hashes when it is made. It was not made under this entry. This entry
changes no hashed file.

---

## DEV-2 (2026-09-28): `experiments/redteam/rts.py` passes `--allow-missing` to the typed clean Layer-B run

**Where.** Prereg §13 item 4.2–4.3 and §14.5 step 5: RT-S (`rts.py et`, `clean-readers`, `sample`, `nodef-clean`,
`stages`, `run`, `score`). All of these build the typed clean Layer-B run of MAIN on W_clean through `rts.py`
`layer_b()`.

**What happened.** This is the downstream consequence that DEV-1 named. The registered first step,
`rts.py et`, ran with the unmodified hashed file (SHA-256 `6da273bd…`) and failed with the input gate's refusal:
`InputError: Layer-B inputs are incomplete (waive a check with DecisionConfig.allow_missing): coverage:
['f_imp@2026-01-02', 'f_cont@2026-01-02', 'f_rev@2026-01-02', 'f_sur@2026-01-02', 'f_net@2026-01-02'] ...`.
These are the same seven N23 flags in the same quiet New Year week as DEV-1. Nothing was written. `layer_b()` built
`DecisionConfig.from_yaml(gen_id=..., name="primary")` with no waiver. The driver's `--allow-missing` option reached
only the reader runs (`stages.reader_log`), so the registered driver could not run on W_clean at all.

**Fix (minimal; one line of logic plus the option's help text).** `layer_b()` now passes the driver's own
`--allow-missing` value to the typed configuration:

```
-    cfg = DecisionConfig.from_yaml(gen_id=w.gen_dir.name, name="primary")
+    cfg = DecisionConfig.from_yaml(gen_id=w.gen_dir.name, name="primary", allow_missing=tuple(a.allow_missing))
-    ap.add_argument("--allow-missing", default="", help="Layer-B input checks waived for the reader runs")
+    ap.add_argument("--allow-missing", default="", help="Layer-B input checks waived for the typed and reader runs")
```

Every RT-S call after this fix runs with `--allow-missing coverage`, the waiver DEV-1 logged for the primary cell.

**Effect.**
- **Decisions.** None change (DEV-1: `allow_missing` is read only by the input gate and the trial record). The typed
  run is `register=False, write=False` in this driver, so no trial row and no run directory come from it.
- **Configuration.** With `--allow-missing coverage` the typed configuration's hash is
  `4ea23931a045dd49a3f22640ca83c51cf262bba1a812e0f685c76aea990b9273`, exactly the registered primary run's (DEV-1).
  Without the option it is `2052f09a…`, as before the fix, so behaviour without the flag is unchanged.
- **Check.** E_t from the fixed driver (3,390 stock-weeks in 113 learned weeks, R^T ≤ 30; `results/redteam/rts/E_t.json`
  SHA-256 `db1b8d40…`) equals the non-cold R^T ≤ 30 rows of the registered primary run's `predictions.parquet`
  (`results/eval/4ea23931a045dd49/W_clean__80d81f4a5566/`) exactly (3,390 = 3,390, same set).
- **Sample sizes, seeds, families, readers, stages, sub-caps:** unchanged.

**Hashes.**

| File | Before | After |
|---|---|---|
| `experiments/redteam/rts.py` | `6da273bd94707cb7bb1cdf4b3de50b58ff272af70d956d9e26dc819ac12e4dd1` | `c88b148747b77af1cf611d3d08427aa042a0cdd78a0c527c2c52ec9a1c36963a` |

`results/analysis/CODE_HASHES.json` is not rewritten: it is an object the lock file hashes. So
`code_hashes.py --check` now lists `experiments/redteam/rts.py` as stale, and this entry is the record of why.
`configs/prereg_v1.yaml`, the pre-registration and every other hashed file are unchanged.

---

## DEV-3 (2026-09-28): `experiments/faith/faith.py t2` gets the same `--allow-missing` pass-through

**Where.** Prereg §9 (FAITH) and §13 item 4.5: "T2 only if its post-unlock dry run fits". `faith.py t2` builds the
typed clean Layer-B run of MAIN on W_clean, draws T2's registered sample from it (`faith.sample_t2`), dry-runs the
exact requests and preflights them.

**What happened.** This is the same failure as DEV-2, in a third driver. The registered call `faith.py t2` ran with the
unmodified hashed file (SHA-256 `75c4a89a…`) after T3 had finished. It failed before any sample, request or
projection with the same gate refusal: `InputError: ... coverage: ['f_imp@2026-01-02', 'f_cont@2026-01-02', ...]`.
Nothing was written and $0 was spent. `cmd_t2` built `DecisionConfig.from_yaml(gen_id=MAIN.name, name="primary")`,
and the `t2` sub-command had no waiver option.

**Fix (minimal).** The `t2` sub-command gets the `--allow-missing` option that `run_layer_b.py` and `rts.py` already
have. `cmd_t2` passes it to the typed configuration:

```
+    allow = tuple(x for x in a.allow_missing.split(",") if x)
-    cfg = DecisionConfig.from_yaml(gen_id=MAIN.name, name="primary")
+    cfg = DecisionConfig.from_yaml(gen_id=MAIN.name, name="primary", allow_missing=allow)
     ...
     p = sub.add_parser("t2")
     p.add_argument("--run", action="store_true")
     p.add_argument("--n", type=int, default=0)
+    p.add_argument("--allow-missing", default="", help="Layer-B input checks waived for the typed clean run")
```

T2 then runs as `faith.py t2 --allow-missing coverage [--run]`.

**Effect.**
- **Decisions, sample, arms, seeds, sizes, the preflight rule:** unchanged. The waiver never enters a decision
  (DEV-1). The typed configuration is `4ea23931…`, the registered primary run's.
- **FAITH hash.** Unchanged: `faith_sha256()` covers `src/jevtrader/redteam/faith.py` and the frozen arms, not this
  driver. It is still `0b64b1ae6866fadd682cc4e7b5b67b69367899cb8dc1da381de31618d92a48fc`.
- Without the option, behaviour is as before the fix.

**Hashes.**

| File | Before | After |
|---|---|---|
| `experiments/faith/faith.py` | `75c4a89a685df56ca8cff8a21010a7ffd818fbcd0943f490bb8e3e12bb6ce570` | `0511464c4b1651b4bef6afe03610be35499537f640e0c902ffae1e2d7491c799` |

As with DEV-2, `CODE_HASHES.json` is not rewritten, so `code_hashes.py --check` lists this file as stale.

---

## DEV-4 (2026-09-29): `experiments/analysis/pd0_operations.py` counts an S1 reader's stock-weeks from its feature rows

**Where.** Prereg §4.1 PD0 ("operating profile per question program and reader") and §14.5 step 7; Table 1's comparator
rows (holistic, holistic + guards, holistic + marking, TA-style, Mono-CoT) and Table 2's per-reader cost columns.

**What happened.** PD0 was called with the comparator readers that Table 1 registers, through the driver's own
repeatable `--reader LABEL=dir` option:
`pd0_operations.py --reader TA-lite=results/gens/tal_ta-lite_wclean-last60_s30 --reader H=results/judgments/h_emu_wclean_et
--reader H-guard=results/judgments/hguard_emu_wclean_et --reader H-spot=results/judgments/hspot_emu_wclean_et
--reader Mono-CoT=results/judgments/mcot_mono-cot_wclean-last60_s30`.
With the unmodified hashed file (SHA-256 `3e5419b6…`) it failed with `ZeroDivisionError: float division by zero` at
the ledger's per-stock-week division, before anything was written. `typed_profile` counts a typed generation's
stock-weeks as its (t, symbol) pairs with a technical (T) request. The H and H-spot clean runs on E_t (P-1) are S1
reader generations with only `H` / `H-spot` requests and no T request, so their stock-week count was 0. $0 spent.

**Fix (minimal; two lines).** When a generation has no T request and has a `features.parquet`, its stock-weeks are
the (t, symbol) pairs of its feature rows. That is the reader's own coverage: one `*_outlook` row per E_t stock-week,
3,390 for H and for H-spot, including stock-weeks with no news item, where the reader sends no request.

```
     tt = req[req["program"] == "T"]
     sw = int(tt[["t", "symbol"]].drop_duplicates().shape[0])
+    if not sw and (gen / "features.parquet").exists():  # an S1 reader run on E_t (H, H-spot): its feature rows
+        sw = int(pd.read_parquet(gen / "features.parquet", columns=["t", "symbol"]).drop_duplicates().shape[0])
```

**Effect.**
- Operational metrics only. PD0 reads no return, label or learned quantity, so the choice was not informed by any
  W_clean outcome.
- Typed generations (MAIN, PRE) always have T requests, so their profile is unchanged. MAIN: 11,600 stock-weeks, as
  before.
- H and H-spot now have 3,390 stock-weeks, so 0.912 requests per stock-week (3,092 requests). Counting only the pairs
  with a request would have given 3,092 and 1.0.
- H-guard has no generation tables (its clean run did not complete; postunlock_paid §1.3), so it stays `not run`.

**Hashes.**

| File | Before | After |
|---|---|---|
| `experiments/analysis/pd0_operations.py` | `3e5419b6d727efb5ff2a13e8db2c924448b1f7531b8d98fed746896e0cbad56b` | `26d6269b2156f8620e0eb75edfc684f621ca7612684c1086cfccbf627bba616e` |

`CODE_HASHES.json` is not rewritten (it is hashed by the lock object), so `code_hashes.py --check` lists this file as
stale. `results/analysis/pd0.json` records the new script and closure SHA-256 in its `meta`.

---

## DEV-5 (2026-09-29): `experiments/analysis/frontier.py` gets `--allow-missing` and `--reader-gens`

**Where.** Prereg §4.2 PRICE, §5 "Fig. 2 frontier replays", §4.1 Table 2 (clean selection return per reader) and
§4.3 (the clean-IC TOST); §14.5 step 7 `frontier.py`.

**What happened.** Two problems, both found before `frontier.py` produced any output. Nothing was read from a
frontier replay when the fix was chosen.
1. **Coverage gate (the DEV-1 week).** Line 286 built `DecisionConfig.from_yaml(..., name="primary")` with no waiver,
   and the driver had no waiver option. So the registered call would meet the refusal that DEV-1, DEV-2 and DEV-3
   met. `postunlock_core.md` §7 item 1 named this in advance.
2. **The registered H clean run is invisible to the driver.** The holistic reader's replays run only
   `if "h_outlook" in set(inputs.raw["feature"])`, and `clean_by_reader` does the same per reader, but `inputs`
   holds only MAIN's generation. The registered clean runs of H and H-spot on E_t (P-1) were written by the hashed
   `rts.py clean-readers` into their own generations, `results/judgments/h_emu_wclean_et/` and
   `hspot_emu_wclean_et/`. The hashed `rts.py` scores these readers by stacking each reader generation into the
   typed inputs (`redteam.stages.stack_raw`, through `stages.reader_log`). Without the same stacking, `frontier.py`
   would silently record the holistic series of Fig. 2 and the H and H-spot rows of Table 2 as "not run", although
   their registered clean runs exist.

**Fix (minimal, opt-in; without the new options the behaviour is unchanged).**

```
+    ap.add_argument("--allow-missing", default="", help="Layer-B input checks waived (as run_layer_b.py)")
+    ap.add_argument("--reader-gens", nargs="*", default=[],
+                    help="clean reader generations stacked into the inputs (P-1 runs on E_t; redteam.stages.stack_raw)")
 ...
-    base = DecisionConfig.from_yaml(LEARNING_YAML, SYSTEM_YAML, gen_id=a.gen_id, name="primary")
+    allow = tuple(x for x in a.allow_missing.split(",") if x)
+    base = DecisionConfig.from_yaml(LEARNING_YAML, SYSTEM_YAML, gen_id=a.gen_id, name="primary", allow_missing=allow)
     inputs = load_inputs(base, start, end, root, data_root=C.data_root(guard))
+    if a.reader_gens:
+        from jevtrader.redteam.stages import stack_raw
+
+        inputs = dataclasses.replace(inputs, raw=stack_raw(inputs.raw, [Path(g) for g in a.reader_gens],
+                                                           inputs.industry or {}))
 ...
-                   inputs={"judgment features": root / "results/judgments" / a.gen_id / "features.parquet"}, argv=argv)
+                   inputs={"judgment features": root / "results/judgments" / a.gen_id / "features.parquet",
+                           **{f"reader features {Path(g).name}": Path(g) / "features.parquet" for g in a.reader_gens}},
+                   argv=argv)
```

Call: `frontier.py --allow-missing coverage --reader-gens results/judgments/h_emu_wclean_et
results/judgments/hspot_emu_wclean_et`. H-guard has no clean generation (postunlock_paid §1.3), so it is not
passed, and `clean_by_reader` records it as "not run".

**Effect.**
- **Decisions.** The waiver changes none (DEV-1). `stack_raw` adds only the rows of features that MAIN does not
  have (`h_outlook`, `hspot_outlook`). They are covered only on E_t, and no configuration other than
  V-reader-H / V-reader-H-spot reads them.
- **Check.** The stacked run reproduces the registered outputs exactly:
  - the typed operating point (J = 10, μ = 2) has weekly IC −0.02386491468418668, the registered primary run's
    (`results/eval/4ea23931…/W_clean__80d81f4a5566/`; PD2's recomputation);
  - its in-band ASR is 0.10456 [0.0820, 0.1280] on 1,119 rows, and the holistic reader's is 0.000894 [0, 0.0028].
    Both equal `h1.json` `pooled`, which `rts.py score` wrote with its own per-reader stacking;
  - typed minus code-only weekly IC is −0.015504066668040896, H3's registered estimate.

  So the stacking changes nothing for the typed configuration, and it reproduces rts.py's H scoring.
- **Replays.** They are unregistered (`register=False, write=False`), as before: no trial row, no run directory.
- **Samples, seeds, sweeps, sizes, clusters:** unchanged.

**Hashes.**

| File | Before | After |
|---|---|---|
| `experiments/analysis/frontier.py` | `740597cfba32bb4eb7f179f90f2bf44d2ccb85aba91989a42cd34ef32eb157ca` | `aa6c99a973e3ad079879db5996ba51ff0823c35738a23caace64ff20f09da1a8` |

As with DEV-2 to DEV-4, `CODE_HASHES.json` is not rewritten, so `code_hashes.py --check` lists the file as stale.

---

## DEV-6 (2026-09-29): H1's realized frame is 1,119 of the 1,400 registered pairs (record; no code change)

**Where.** Prereg §3.1 "Attacks" ("One per unit and family, A1–A7, in the static setting: 1,400 paired outcomes") and
the H1 power block. Audit finding A2 (`docs/reviews/postunlock_audit.md`).

**What happened.** H1 (`experiments/analysis/h1_asr.py` → `results/analysis/h1.json`) scored **1,119 paired
(unit, family) rows in 113 attack-week clusters**, not 1,400. The 281 missing unit-families were skipped, before any
paid call and before any outcome was read, by the hashed planner `redteam.run.plan_batches`
(`results/redteam/rts/typed/skipped.parquet`), for two reasons:

| Reason | Families | Unit-families skipped | Rule |
|---|---|---|---|
| `no_host_item` | A1, A2, A3, A4 (27 each, the same 27 units in every family) | 108 | the edit families rewrite a real news item of the unit's stock-week; 27 of the 200 units had no host item (`inject.NoHostItem`, `run.py` lines 278–283) |
| `not_eligible_at_t_plus_2` | A7 | 173 | "A7 … counts only units that are still eligible at t+2" (prereg §3.1) |

Scored pairs per family (`h1.json` `per_family`): **A1–A4 173 each, A5 200, A6 200, A7 27** (173 × 4 + 200 × 2 + 27 =
1,119).

- The A7 loss is the registered rule. The `no_host_item` rule is in the hashed planner, so it was fixed before the
  unlock and involves no discretion, but the pre-registration text does not state it, and `postunlock_paid.md` §2.3
  attributed all 281 to the A7 rule. This entry corrects that attribution.
- The realized 1,119 is below the registered worst case of 1,200 pairs (`power_prereg.json`
  `H1.registered_if_A7_units_all_lost`), because that worst case assumed no edit-family loss.

**MDE at the realized frame.** By the registered formula (`experiments/prereg/power_prereg.py` `h1_block`:
z_holm·√(2π_d/n), z_holm = 2.128 + 0.8416), at n = 1,119: **0.056 at π_d = .2 and 0.079 at .4** (0.040 / 0.069 at
π_d = .1 / .3), against the registered 0.050–0.071 at 1,400. (For the C1-secondary family, z = 3.2916, the MDE at
1,119 is 0.062–0.088 against the registered 0.056–0.079.)

**Effect.**
- The confirmatory statistic, its test and its Holm step are unchanged; they already ran on the 1,119 pairs.
- Reporting: the paper gives both frames. `\nPairedAttacks` now prints the realized 1,119 and
  `\nPairedAttacksRegistered` the registered 1,400; `\hOneMDELow`/`\hOneMDEHigh` print the MDE at the realized pairs
  (0.056/0.079) and `\hOneMDELowRegistered`/`\hOneMDEHighRegistered` the registered 0.050/0.071. The code change that
  does this is DEV-9.

**Code hashes.** None changed by this entry.

---

## DEV-7 (2026-09-29): registered items that did not complete or were not run, with their status (record; audit A3, A7)

**Where.** Prereg §16 ("any change is … logged … with its reason, date and effect"). Audit finding A3: DEV-1..DEV-5
logged code and waiver changes only, not the registered runs and analyses that did not complete or were not run.
Statuses below are as of the end of 2026-09-29 (this entry is updated in place only by appending a dated line).

| Registered item (prereg §) | Status | Effect on the paper |
|---|---|---|
| H-guard clean on E_t (§13 item 4.2) | 2026-09-28: **not completed ($0)**, refused at start by the in-process guard. 2026-09-29: **completed** under DEV-8 ($0.6047, inside the $2.00 sub-cap; `results/judgments/hguard_emu_wclean_et/`) | the C1-secondary contrast typed vs H-guard, V-reader-H-guard and the H-guard rows of Tables 1 and 2 are now produced by their registered entry points |
| Reverted edit to `src/jevtrader/redteam/projection.py` (2026-09-28) | change-control event: a draft that skipped guard reservations for response-cache hits; refused by the permission check as a weakening of a spend guard; **reverted at once**; the file's SHA-256 is back to `9279cf8c261359e1b74116f183d1590f042e820fdb00b6a7053b4d7b314ad863` (checked 2026-09-29); **no run used the edited file** | none |
| V-reader-H, V-reader-H-spot, V-reader-H-guard trial rows (§4.4, §5) | were missing (no registered entry point; postunlock_results §6 item 2); **registered under DEV-10** | PD1, the DSR and Table 4 run under the registered strict rule |
| PD1 / Table 4 / DSR / Random-20 (§4.1, §4.4) | were blocked by the registered strict trial rule (missing V-reader-H/-H-spot/-H-guard rows). **Run** on 2026-09-29 once DEV-10 registered those rows, with its registered defaults and no code change (`pd1_portfolio.py` SHA-256 `94974995…`, as registered): `results/analysis/pd1.json`, Table 4 (`paper/tables/table4_portfolio.tex`) | Table 4 and the caption macros are filled |
| PD3 all-stores-off (§4.1, "always reported") | **not run** in registered form: no hashed builder of the stores-off Layer-B replay over MEM existed at the commit (precommit_report §4 item 6). `results/analysis/pd3.json` says "no all-stores-off replay (MEM not run)"; the accurate reason is "no stores-off replay builder" (MEM's two arms were generated). An **exploratory** replay (`experiments/exploratory/pd3_stores_off.py` → `results/exploratory/pd3_stores_off.json`) is reported in App. "Exploratory analyses", labelled not pre-registered | Table 3's all-stores-off cell stays "---" |
| X6-NG (§9, §10) | not run: no entry point | the S1-flagged-share statement is not made |
| NG-D1 (§8, §9) | not run: no entry point | D1 is reported for emu only |
| NG-FID determinism (§9 NG extras) | not run: no entry point | — |
| B11 coverage placebo (§3.3, §6) | not run: no hashed B11 Layer-B run and no `--coverage-placebo` in the registered H3 call | `\hThreePlaceboDeltaIC` stays a placeholder; Table 3 cell "---" |
| B8 logistic baseline (§6, "CPU, after the unlock") | no registered output: B8 is not a PD1 row (§4.1) and no registered entry point writes it. An exploratory row exists (`pd1_extended.py`, App. "Exploratory analyses") | stated as not pre-registered |
| D4 break test (§8, "$0, after the unlock") | not run: no entry point | the paper states D1–D3 only |
| AgentDojo triad (RB-4) and the RB-16 checklist (§3.1 "Also reported with CIs") | not run as such: `h1.json` has signed harm and ASR, but no entry point computes the triad or fills the checklist | App. "Attack families and reporting checklists" stays a stub |
| VC-14 beyond the sentinel: F-B reading accuracy, T answers vs hand-rule twins, policy s^pol in high-intensity weeks (§3.2) | not run: no entry point; only `h2.json agent_level.sentinel_h_vs_T3` exists | App. "Program-level validity" reports the sentinel row only |
| MEM-9 / A7-para and A7 persistence at t+3..t+4 (§4.1) | not run: the hashed `redteam.stages.STAGES` has no A7-para or persistence stage, although RT-S had $7.25 of headroom after stage 5; `h1.json persistence` is empty | no persistence statement |
| FAITH T1 over all cards, T5 (§9) | not run: no entry point (T1 exists only for the Fig. 3 card, `fig3_card.json`) | — |
| FAITH T4-TAL, T4-H (§9) | not run (budget), as registered | — |
| RT-S stage 3, TA-lite under attack (§10) | not built, as registered; the contrast enters the C1-secondary Holm with p = 1 | — |
| FIDX, CF (§13 item 4.7) | not run (optional; skipped by the orchestrator) | `\confOrderShift` stays a placeholder |
| OPS-3 replay determinism, FID V1/V2 summaries, share of D_t, anchor rates, reachable-byline ASR, TA-lite/Mono-CoT clean selection return (§14.3) | not run: registered as placeholders without a producer | macros stay placeholders |
| LIVE (§13 item 4.1) | pending: calendar-bound (10-02, 10-09, 10-16) | deployment rows and Fig. 3's registered card pending |

**Audit A7 (property tests not hashed at the commit).** `tests/test_band_guard_bound.py` and
`tests/test_bound_history.py`, which gate BOUND through `bound_claim.py`, are in neither `CODE_HASHES.json` nor
`results/prereg/hashes.json`. Their current SHA-256 are `15c4ce3a6ed838c0e2585b33e889acdaa69433ceb60affa55c85d1d175037fa9`
and `fb22d7afd4d2b67eb286cb21aa61d791aa7862b6abd5c359f675952d7028ec63`, equal to the values `bound.json`
(`property_tests.files`) recorded at run time on 2026-09-28; their mtimes (2026-09-27 18:56 and 2026-09-26 05:16,
local time) predate the commit. Nothing suggests a change, but it cannot be proved from the registration objects; this
entry records the gap. No file changed.

**Code hashes.** None changed by this entry.

---

## DEV-8 (2026-09-29): `experiments/redteam/rts.py clean-readers` gets `--max-parallel-units` (H-guard clean on E_t)

**Where.** Prereg §13 item 4.2 ("H, H-spot, H-guard clean on E_t … each only if its own preflight fits") and §3.1
C1-secondary (typed vs H-guard). `postunlock_paid.md` §1.3; audit A3.

**What happened (2026-09-28).** H-guard's registered clean run fitted its preflight ($0.64 high / $0.56 point against
the $2.00 HGUARD sub-cap) but was refused at start by the in-process spend guard, before any paid call:
`BudgetExceeded: stage guard: spent $0.0000 + reserved $1.9998 + this call $0.0003 > $2.0000`. The guard
(`redteam.projection._Guarded.ask`) reserves `estimate_cost` for every call before the backend looks in its response
cache. H-guard's generation also runs N01, about 110,000 requests that are all free MAIN cache hits, and with the
spec's default `max_parallel_units = 16` thousands of them were in flight at once: their reservations alone filled the
headroom. A fix that skipped reservations for cache hits was drafted on 2026-09-28, refused by the permission check as
a weakening of a spend guard, and reverted (see DEV-7); it was not used.

**Fix (minimal; opt-in; the guard is not changed).** The paid run of `clean-readers` can run fewer Layer-A units at
once. The option only sets `GeneratorSpec.max_parallel_units`, which is not part of any unit fingerprint, request,
answer, cache key or cap; the dry run and the preflight are unchanged.

```
-        gen = _generate(dataclasses.replace(spec, out_root=ROOT / "results" / "judgments"), s1)
+        par = {"max_parallel_units": a.max_parallel_units} if getattr(a, "max_parallel_units", 0) else {}
+        gen = _generate(dataclasses.replace(spec, out_root=ROOT / "results" / "judgments", **par), s1)
 ...
     p.add_argument("--reader", default="")
+    p.add_argument("--max-parallel-units", type=int, default=0,
+                   help="paid run only: fewer concurrent Layer-A units (0 = the spec's default; the guard is unchanged)")
```

Why 1: the guard's reservation for one week's N01 unit is at most about $0.97 by the same `estimate_cost` (the
largest week in the H-guard dry run; median $0.44), so one unit at a time keeps reservations plus spend below the
$2.00 headroom, while the ledger's own sub-cap and hard-cap checks stay in force underneath.

**Run (2026-09-29).**
- Free preflight first: `rts.py --allow-missing coverage clean-readers --reader H-guard` →
  `PREFLIGHT HGUARD: projected $0.61 high (point $0.53) vs sub-cap $2.00 (run spent $0.00, headroom $2.00); ledger
  $194.38 of hard cap $245.00 -> OK` (`results/redteam/rts/clean_readers_hguard_preflight_only.json`).
- Paid: `rts.py --allow-missing coverage clean-readers --reader H-guard --run --max-parallel-units 1` (it repeats its
  dry run and preflight, then pays under the guard of $2.00). The run spent **$0.6047** (ledger: 1,550 `HGUARD/*` rows: H-guard $0.5701, N1 $0.0345 including retries), inside the $2.00 sub-cap and below the $0.61 high projection. The guard never refused a call. 112,853 requests (111,339 cached; 1,489 paid H-guard requests, 25 paid N1), 226 units, validity 0.99995 (16 N1 answers degraded after 3 attempts, which abstain), generation wall time 510 s. Output: `results/judgments/hguard_emu_wclean_et/` with 3,390 `hguard_outlook` rows, one per E_t stock-week (MANIFEST `spend_usd` 0.6047). Ledger after: **$194.9843** (625,462 rows). Logs: `results/redteam/rts/logs/rts_clean_readers_hguard_{dry,run}_20260929.log`.

**Effect.**
- Answers, sample (E_t), programs, bank, backend and caps: unchanged. Only the concurrency of the paid run changed.
- `clean_readers.json` is rewritten by the driver with the H-guard record only; the H and H-spot records of
  2026-09-28 are in `clean_readers_preflight_only.json` (identical to the file before this run) and in their
  generations' MANIFESTs.

**Hashes.**

| File | Before | After |
|---|---|---|
| `experiments/redteam/rts.py` | `c88b148747b77af1cf611d3d08427aa042a0cdd78a0c527c2c52ec9a1c36963a` (after DEV-2) | `f3324983935b848908e2c102c63bdb5e81eef1a8db042ef03a3ade0074185be4` |

`src/jevtrader/redteam/projection.py` (the guard) is unchanged: `9279cf8c261359e1b74116f183d1590f042e820fdb00b6a7053b4d7b314ad863`.


<details><summary>diff</summary>

```diff
--- a/experiments/redteam/rts.py
+++ b/experiments/redteam/rts.py
@@ -204,7 +204,8 @@
             rec["aborted"] = True
             return rec
         s1 = P.paid_guard(paid_backend(), pf.headroom)
-        gen = _generate(dataclasses.replace(spec, out_root=ROOT / "results" / "judgments"), s1)
+        par = {"max_parallel_units": a.max_parallel_units} if getattr(a, "max_parallel_units", 0) else {}
+        gen = _generate(dataclasses.replace(spec, out_root=ROOT / "results" / "judgments", **par), s1)
         rec["gen_dir"] = str(gen)
         rec["guard_spent"] = s1.spent
     return rec
@@ -472,6 +473,8 @@
     p = sub.add_parser("clean-readers")
     p.add_argument("--run", action="store_true")
     p.add_argument("--reader", default="")
+    p.add_argument("--max-parallel-units", type=int, default=0,
+                   help="paid run only: fewer concurrent Layer-A units (0 = the spec's default; the guard is unchanged)")
     p = sub.add_parser("nodef-clean")
     p.add_argument("--run", action="store_true")
     sub.add_parser("sample")
```
</details>

---

## DEV-9 (2026-09-29): `experiments/report/make_numbers.py`: descriptive H1 wording for a reversed estimate, realized H1 frame, p-value format, PD2 sentence, source labels; prereg §3.1 wording deviation

**Where.** Prereg §3.1 "Registered result sentences", §3.4 (sentences generated by `make_numbers.py`), §4.2 (claim
register: H1, PD2). Audit findings A1, A2, A4, A5, A6 (`docs/reviews/postunlock_audit.md`). The pre-registration file
itself is **not** edited (its SHA-256 stays `b8bf8cf6…84b9a`); this entry is the record of the changed wording, and the
paper uses the wording below.

**1. H1 (audit A1): a third, descriptive wording.** The registration fixed a "stated" sentence (Holm-adjusted p < .05)
and a "null" fallback ("… did not differ detectably (MDE ≈ …)"), but no sentence for an estimate whose 95% CI excludes
0 on the side **opposite** to the registered claim. That is the registered outcome: ASR_typed − ASR_holistic = +0.1037,
95% CI [0.0810, 0.1273], one-sided p (typed < holistic) = 1.0, Holm-adjusted p = 1.000 (`h1.json`,
`confirmatory.json`). The registered fallback is false on that outcome.

*Prereg §3.1, deviated text (added as a third case; the stated and fallback sentences are unchanged):*
> **reversed** (Holm-adjusted p ≥ .05 **and** the 95% CI of ASR_typed − ASR_holistic lies entirely above 0):
> "At the tested attacker and budget, the pre-registered one-sided test that per-item guarded decomposition lowers
> attack success relative to the holistic reader was not supported (Holm-adjusted p = [p]); the estimate points the
> other way, with attack success of [ASR typed]% for per-item guarded decomposition and [ASR holistic]% for the
> holistic reader (difference [Δ], 95% CI [CI])." The two registered clauses (bound; localization) are inserted before
> its final period under their registered conditions, exactly as for the fallback.
>
> Abstract clause in that case: "per-item guarded decomposition does not lower attack success versus a holistic
> reader ([ASR typed]% vs. [ASR holistic]%)".

- The wording is **descriptive**: the registered test's own Holm-adjusted p, the two ASRs, the paired estimate and its
  registered percentile CI. It carries no new p-value and no new test; a reversal is not a registered hypothesis and
  is not presented as a finding of one.
- Condition in code: `Sentence.reversed` / `reversal` (`ci_opposite("h1", ("confirmatory", "ci"), "below")`: the
  registered claim is estimate < 0, so "opposite" is CI lower bound > 0). The switch is mechanical, from registered
  outputs only.
- Generated result (numbers.tex): `\hOneSentence` = "At the tested attacker and budget, the pre-registered one-sided
  test that per-item guarded decomposition lowers attack success relative to the holistic reader was not supported
  (Holm-adjusted $p = 1.000$); the estimate points the other way, with attack success of 10.5% for per-item guarded
  decomposition and 0.1% for the holistic reader (difference 0.104, 95% CI [0.081, 0.127]); the influence bound capped
  the damage for every reader, and only per-item readers localized the planted item." `\hOneAbstract` = "per-item
  guarded decomposition does not lower attack success versus a holistic reader (10.5% vs. 0.1%)".

**2. Realized H1 frame (audit A2; DEV-6).** `\nPairedAttacks` prints the realized pairs (`h1.json`
`confirmatory.n_pairs` = 1,119); the new `\nPairedAttacksRegistered` prints the registered 1,400. `\hOneMDELow` /
`\hOneMDEHigh` print the MDE at the realized pairs by the registered formula (`power_prereg.h1_block(1119)`: 0.056 /
0.079); the new `\hOneMDELowRegistered` / `\hOneMDEHighRegistered` print the registered 0.050 / 0.071.

**3. p-value format (audit A4).** A p-value below 0.001 prints as m × 10^e with two significant digits instead of
rounding to "0.000": `\hTwoPholm` = 3.3 × 10^-20 (Holm-adjusted 3.2917e-20). Values ≥ 0.001 print as before.

**4. PD2 (audit A5).** New macros `\poolLogLossDiff` (−0.00265), `\poolLogLossDiffCI` ([−0.00453, −0.00077]) from
`pd2.json` `paired.primary_vs_V-EW-log.delta_log_loss_b_minus_a`, and a generated sentence `\poolSentence` with the
§4.2 claim-register logic: stated if `claim_composition_adds` is True; the registered retirement wording "no
detectable difference from equal-weight pooling" if the CI covers 0; and, because that wording is false when the CI
lies entirely below 0 (the registered outcome), the descriptive "The learned pool is detectably worse than the
equal-weight log pool in log loss (weekly difference, equal-weight minus learned, −0.00265, 95% CI [−0.00453,
−0.00077])." Estimate and CI only.

**5. Labels and source comments (audit A6).** `\hTwoSpearmanVerb` / `\hTwoSpearmanNG` are described as POOLED
Spearman values with week-block CIs (not weekly means); a new `\hTwoSpearmanEmuPooled` (0.122) gives emu's value of the
same pooled statistic; `\valueTextCI` is described as a two-sided NW(4) interval; the abstract clauses cite their real
source (`confirmatory.json claims.H*` through the full sentence), not the non-existent `results/c1|c2|c3/summary.json`.

**Effect.** No registered number, test, Holm decision or clause condition changes. `make_numbers.py` now writes
157 filled macros and 33 placeholders (before this task: 146 / 37; the new macros are `nPairedAttacksRegistered`, `hOneMDELowRegistered`, `hOneMDEHighRegistered`, `hTwoSpearmanEmuPooled`, `poolLogLossDiff`, `poolLogLossDiffCI`, `poolSentence`, and PD1's `nTrials`, `dsr`, `dsrNhat` and `randomPct` are now filled from the registered `pd1.json`). Every change is presentation of registered outputs; the reversed wordings apply only when their
registered outputs say so.

**Consequential test update (not hashed).** `tests/test_analysis_report.py::test_make_numbers_protocol_constants_match_the_preregistration` asserted the old meaning of `hOneMDELow/High` and `nPairedAttacks`; it now checks the registered values on the `*Registered` macros and the realized ones on the originals, and a new `test_make_numbers_reversed_wording_switch` tests the switch and `fmt_p` (SHA-256 `124e997ead64dc198f409e1918407845a7951f608aab3f648af07c06b5b991e3` → `8717f4421b12141612de0ca05ab1649e9b2770fe39c91d63b25b0d11567c2c3e`).

**Hashes.**

| File | Before | After |
|---|---|---|
| `experiments/report/make_numbers.py` | `754c41d6d74e1c594a583b85f5be3913be7e861f7172895737b01fc61816c1b9` | `22a91accd8ca937f008565d8103b2862b85f3a703016eaa1639cc66721582f24` |

The full change, as a unified diff of the file (reversing it on the current file reproduces the "before" hash):

<details><summary>diff</summary>

```diff
--- a/experiments/report/make_numbers.py
+++ b/experiments/report/make_numbers.py
@@ -157,9 +157,24 @@
     ref: str = ""
     confirmed_clauses: list = field(default_factory=list)  # Clause, appended to `confirmed` when their cond holds
     fallback_clauses: list = field(default_factory=list)  # Clause, appended to `fallback` when their cond holds
+    # DEV-9 (docs/design/prereg_deviations.md): descriptive wording used instead of `fallback` when the registered
+    # one-sided test did not reject AND the estimate's 95% CI excludes 0 on the side OPPOSITE to the registered claim
+    # (`reversal` returns True). It states the estimate and its CI only (no new p-value); fallback clauses still apply.
+    reversed: str = ""
+    reversal: Callable[[], bool | None] | None = None
+
+    def is_reversed(self, outcome: bool) -> bool:
+        if outcome or not self.reversed or self.reversal is None:
+            return False
+        try:
+            return self.reversal() is True
+        except Exception:  # noqa: BLE001 -- a malformed source never switches the wording
+            return False
 
     def compose(self, outcome: bool) -> str:
         base, clauses = (self.confirmed, self.confirmed_clauses) if outcome else (self.fallback, self.fallback_clauses)
+        if self.is_reversed(outcome):
+            base = self.reversed
         extra = []
         for c in clauses:
             try:
@@ -346,12 +361,18 @@
         Num("nDemotionUnits", "demotion units (plus an anchor stratum)", PREREG),
         Num("nAdaptiveItems", "items in the adaptive arm", PREREG),
         Num("maxAdaptiveQueries", "black-box query budget per item", PREREG),
-        Num("nPairedAttacks", "paired attacks in H1 (units x families)", PREREG),
+        Num("nPairedAttacks", "paired attacks scored in H1 (realized frame, DEV-6/DEV-9; registered frame: "
+            "nPairedAttacksRegistered)", PREREG),
+        Num("nPairedAttacksRegistered", "paired attacks registered for H1 (units x families; prereg §3.1)", PREREG),
         Num("nPropertyCases", "randomized cases per property test (at least)",
             "tests/test_band_guard_bound.py N_CHUNKS x PER_CHUNK (test_proposition_randomized)",
             get=test_case_count("tests/test_band_guard_bound.py", "N_CHUNKS", "PER_CHUNK")),
-        Num("hOneMDELow", "H1 MDE, low end", PREREG + " power.C1_MDE_n2800_pid0.2"),
-        Num("hOneMDEHigh", "H1 MDE, high end", PREREG + " power.C1_MDE_n2800_pid0.4"),
+        Num("hOneMDELow", "H1 MDE at the realized pairs, low end (pi_d = .2; registered formula; DEV-6/DEV-9)",
+            PREREG + " power.C1_MDE_n2800_pid0.2"),
+        Num("hOneMDEHigh", "H1 MDE at the realized pairs, high end (pi_d = .4; registered formula; DEV-6/DEV-9)",
+            PREREG + " power.C1_MDE_n2800_pid0.4"),
+        Num("hOneMDELowRegistered", "H1 MDE at the registered 1,400 pairs, low end (pi_d = .2; prereg §3.1)", PREREG),
+        Num("hOneMDEHighRegistered", "H1 MDE at the registered 1,400 pairs, high end (pi_d = .4; prereg §3.1)", PREREG),
         Num("hTwoMDELow", "H2 MDE, low end", PREREG + " power.C2_MDE_between_sd0.0"),
         Num("hTwoMDEHigh", "H2 MDE, high end", PREREG + " power.C2_MDE_between_sd0.05"),
         Num("hThreeMDELow", "H3 MDE, low end (IC)", PREREG + " power.C3_MDE_rho0.8"),
@@ -426,7 +447,8 @@
             "sizing (bps)", "results/bound/clean_costs.json (V-size-v1)", "Fig. 2 caption"),
         Num("valueText", "value of the text: paired mean weekly selection return of the signal book, J = inf minus "
             "J = 0 (bps per week)", "results/eval/frontier.json (V-J0, V-Jinf replays)", "www M9"),
-        Num("valueTextCI", "week-block CI of valueText (formatted [lo, hi])", "results/eval/frontier.json", "www M9"),
+        Num("valueTextCI", "two-sided Newey-West(4) 95% CI of valueText (formatted [lo, hi]; frontier.py paired_bps -> "
+            "stats.nw_test; not a week-block bootstrap; audit A6)", "results/eval/frontier.json", "www M9"),
         Num("priceBand", "price of the band at the operating point: selection return at J = inf minus J (bps per week)",
             "results/eval/frontier.json", "www M9"),
         Num("priceSharePct", "priceBand as a share of valueText (%)", "results/eval/frontier.json", "www M9"),
@@ -454,8 +476,12 @@
         Num("hTwoCIEmu", "H2 95% CI, emu", "results/c2/emu/summary.json"),
         Num("hTwoPholm", "H2 Holm-adjusted p", "results/c2/emu/summary.json"),
         Num("hTwoPlaceboEmu", "H2 placebo Spearman (AR[-5,-1]), emu", "results/c2/emu/summary.json"),
-        Num("hTwoSpearmanVerb", "H2 weekly Spearman, VERB (descriptive)", "results/c2/verb/summary.json"),
-        Num("hTwoSpearmanNG", "H2 weekly Spearman, NG (descriptive)", "results/c2/ng/summary.json"),
+        Num("hTwoSpearmanVerb", "H2 per implementation, VERB: POOLED Spearman over its 2,000 units with a week-block CI "
+            "(descriptive; not the weekly mean of hTwoSpearmanEmu; audit A6)", "results/c2/verb/summary.json"),
+        Num("hTwoSpearmanNG", "H2 per implementation, NG: POOLED Spearman over its 500 units with a week-block CI "
+            "(descriptive; not the weekly mean of hTwoSpearmanEmu; audit A6)", "results/c2/ng/summary.json"),
+        Num("hTwoSpearmanEmuPooled", "H2 per implementation, emu: POOLED Spearman on the full frame (the statistic "
+            "of hTwoSpearmanVerb/NG; descriptive; DEV-9)", "results/c2/emu/summary.json"),
         Num("hTwoSpearmanPreOpen", "H2 weekly Spearman, emu, items created before 09:30 ET only (control)",
             "results/c2/emu/summary.json", "www M8"),
         Num("hTwoSpearmanRelease", "H2 weekly Spearman, emu, on W_release (sensitivity)", "results/c2/emu/summary.json",
@@ -468,6 +494,10 @@
         Num("hThreePholm", "H3 Holm-adjusted p", "results/c3/summary.json"),
         Num("hThreePlaceboDeltaIC", "rank-IC increment over the coverage placebo", "results/c3/coverage_placebo.json"),
         Num("poolOracleShare", "share of the oracle gap captured by the learned pool (%)", "results/ma/pools.json"),
+        Num("poolLogLossDiff", "PD2: weekly log-loss difference, V-EW-log minus the learned pool (negative = the "
+            "learned pool has the higher loss; DEV-9, audit A5)", "results/ma/pools.json"),
+        Num("poolLogLossDiffCI", "PD2: NW(4) 95% CI of poolLogLossDiff (formatted [lo, hi]; DEV-9, audit A5)",
+            "results/ma/pools.json"),
     ]),
     Group("Results: traceability and portfolio (§5.5-§5.6)", [
         Num("ledgerResidual", "largest ledger sum residual", "cards.jsonl (T1 exactness)", "Fig. 3 caption"),
@@ -556,7 +586,15 @@
         # their own outputs hold (bound.json via confirmatory.json claims.BOUND; localization.json claim)
         fallback=(r"At the tested attacker and budget, per-item guarded decomposition and the holistic reader did "
                   r"not differ detectably (MDE $\approx$ \hOneMDELow--\hOneMDEHigh)."),
-        ref="prereg §3.1; positioning_v2 §2.7 (reader name per draft-0 editor review §4)"),
+        # DEV-9 (audit A1): the registration has no sentence for an estimate whose CI excludes 0 on the side opposite
+        # to the claim; the null sentence would then be false. Descriptive: the registered test's result (its own
+        # Holm-adjusted p, no new p-value), then the two ASRs and the paired estimate with its 95% CI.
+        reversed=(r"At the tested attacker and budget, the pre-registered one-sided test that per-item guarded "
+                  r"decomposition lowers attack success relative to the holistic reader was not supported "
+                  r"(Holm-adjusted $p = \hOnePholm$); the estimate points the other way, with attack success of "
+                  r"\hOneASRTyped\% for per-item guarded decomposition and \hOneASRHolistic\% for the holistic reader "
+                  r"(difference \hOneDelta{}, \ciLevel\% CI \hOneCI{})."),
+        ref="prereg §3.1; positioning_v2 §2.7 (reader name per draft-0 editor review §4); DEV-9"),
     Sentence(
         "hTwoSentence", "H2 result sentence", "results/c2/emu/summary.json (Holm decision)",
         # registered wording (prereg §3.2); the SEC clause is added only if h2.json's sec_claim holds
@@ -600,29 +638,45 @@
 # the abstract stays within 200 words whatever the outcomes.
 SENTENCES += [
     Sentence(
-        "hOneAbstract", "H1 clause of the abstract", "results/c1/summary.json (Holm decision, as hOneSentence)",
+        "hOneAbstract", "H1 clause of the abstract",
+        "results/analysis/confirmatory.json claims.H1 via hOneSentence (and h1.json confirmatory.ci for DEV-9)",
         confirmed=(r"per-item guarded decomposition lowers attack success from \hOneASRHolistic\% to \hOneASRTyped\% "
                    r"versus a holistic reader"),
         fallback=r"per-item guarded decomposition and a holistic reader do not differ detectably in attack success",
-        outcome=_same_outcome("hOneSentence"), ref="positioning_v2 §6, §2.7"),
+        reversed=(r"per-item guarded decomposition does not lower attack success versus a holistic reader "
+                  r"(\hOneASRTyped\% vs.\ \hOneASRHolistic\%)"),
+        outcome=_same_outcome("hOneSentence"), ref="positioning_v2 §6, §2.7; DEV-9"),
     Sentence(
         "boundAbstract", "reach and price clause of the abstract and conclusion",
-        "as boundSentence (does the valueText CI exclude zero?)",
+        "results/analysis/frontier.json value_of_text.ci_excludes_zero via boundSentence",
         confirmed=(r"attacks use at most \budgetUseMax\% of the bound's reach, and the band gives up \priceSharePct\% of "
                    r"the text's value"),
         fallback=(r"attacks use at most \budgetUseMax\% of the bound's reach, and the text's value is not "
                   r"distinguishable from zero"),
         outcome=_same_outcome("boundSentence"), ref="draft-0 www B2, M9"),
     Sentence(
-        "hTwoAbstract", "H2 clause of the abstract", "results/c2/emu/summary.json (Holm decision, as hTwoSentence)",
+        "hTwoAbstract", "H2 clause of the abstract", "results/analysis/confirmatory.json claims.H2 via hTwoSentence",
         confirmed=r"item judgments track announcement-window abnormal returns (weekly Spearman \hTwoSpearmanEmu{})",
         fallback=r"item judgments do not detectably track announcement-window abnormal returns",
         outcome=_same_outcome("hTwoSentence"), ref="positioning_v2 §6, §2.7"),
     Sentence(
-        "hThreeAbstract", "H3 clause of the abstract", "results/c3/summary.json (Holm decision, as hThreeSentence)",
+        "hThreeAbstract", "H3 clause of the abstract", "results/analysis/confirmatory.json claims.H3 via hThreeSentence",
         confirmed=r"the judgments add \hThreeDeltaIC{} rank-IC over code-only features",
         fallback=r"no rank-IC increment over code-only features is detected",
         outcome=_same_outcome("hThreeSentence"), ref="positioning_v2 §6, §2.7"),
+    # DEV-9 (audit A5): PD2's claim-register wording (prereg §4.2) as a generated sentence. The registered retirement
+    # text ("no detectable difference from equal-weight pooling") is false when the CI lies entirely below 0, so that
+    # case gets a descriptive wording (estimate and CI only).
+    Sentence(
+        "poolSentence", "PD2 composition sentence (prereg §4.2 claim register)",
+        "results/analysis/pd2.json claim_composition_adds",
+        confirmed=(r"The learned pool has lower log loss than the equal-weight log pool (weekly difference, "
+                   r"equal-weight minus learned, \poolLogLossDiff{}, \ciLevel\% CI \poolLogLossDiffCI{})."),
+        fallback=(r"The learned pool shows no detectable difference from equal-weight pooling (weekly log-loss "
+                  r"difference, equal-weight minus learned, \poolLogLossDiff{}, \ciLevel\% CI \poolLogLossDiffCI{})."),
+        reversed=(r"The learned pool is detectably worse than the equal-weight log pool in log loss (weekly "
+                  r"difference, equal-weight minus learned, \poolLogLossDiff{}, \ciLevel\% CI \poolLogLossDiffCI{})."),
+        ref="prereg §4.2 PD2; DEV-9"),
 ]
 
 
@@ -694,11 +748,47 @@
     return get
 
 
+def fmt_p(p: float) -> str:
+    """A p-value: three decimals, or m x 10^e (two significant digits) below 0.001, so that a p-value never prints
+    as 0.000 (DEV-9, audit A4)."""
+    if p >= 0.001:
+        return half_up(p, 3)
+    m, e = f"{p:.1e}".split("e")
+    return f"\\ensuremath{{{m}\\times 10^{{{int(e)}}}}}"
+
+
 def holm_p(label: str) -> Callable[[], str | None]:
     def get():
         fam = _walk(registered_result("confirmatory"), ("family",)) or []
         r = next((x for x in fam if x.get("hypothesis") == label), None)
-        return None if r is None or not _finite(r.get("p_holm")) else half_up(r["p_holm"], 3)
+        return None if r is None or not _finite(r.get("p_holm")) else fmt_p(r["p_holm"])
+    return get
+
+
+def h1_realized_mde(pid: float) -> Callable[[], str | None]:
+    """DEV-6/DEV-9 (audit A2): H1's MDE by the REGISTERED formula (experiments/prereg/power_prereg.py h1_block,
+    z_holm * sqrt(2 pi_d / n)) at the realized number of scored pairs (h1.json confirmatory.n_pairs)."""
+    def get():
+        n = _walk(registered_result("h1"), ("confirmatory", "n_pairs"))
+        if not _finite(n) or n <= 0:
+            return None
+        import importlib.util
+
+        spec = importlib.util.spec_from_file_location("power_prereg", ROOT / "experiments/prereg/power_prereg.py")
+        mod = importlib.util.module_from_spec(spec)
+        spec.loader.exec_module(mod)
+        return half_up(mod.h1_block(int(n))[f"mde_pid{pid}"], 3)
+    return get
+
+
+def ci_opposite(name: str, ci_path: tuple, claim_side: str) -> Callable[[], bool | None]:
+    """DEV-9: True iff the registered output's 95% CI lies entirely on the side OPPOSITE to the registered claim
+    (claim_side "below": the claim is estimate < 0; "above": estimate > 0)."""
+    def get():
+        ci = _walk(registered_result(name), ci_path)
+        if not (isinstance(ci, list) and len(ci) == 2 and all(_finite(x) for x in ci)):
+            return None
+        return bool(ci[0] > 0) if claim_side == "below" else bool(ci[1] < 0)
     return get
 
 
@@ -770,9 +860,16 @@
     "universeSize": ("src/jevtrader/pipeline/inputs.py UNIVERSE_TOP", code_const("src/jevtrader/pipeline/inputs.py",
                                                                                  "UNIVERSE_TOP")),
     "nPromotionUnits": (f"{POWER} H1.registered.units", power("H1", "registered", "units", integer=True)),
-    "nPairedAttacks": (f"{POWER} H1.registered.pairs", power("H1", "registered", "pairs", integer=True)),
-    "hOneMDELow": (f"{POWER} H1.registered.mde_pid0.2", power("H1", "registered", "mde_pid0.2")),
-    "hOneMDEHigh": (f"{POWER} H1.registered.mde_pid0.4", power("H1", "registered", "mde_pid0.4")),
+    # DEV-9 (audit A2): the realized H1 frame; the registered values keep their own macros
+    "nPairedAttacks": (f"{ANALYSIS}/h1.json confirmatory.n_pairs (realized; DEV-6)",
+                       res("h1", "confirmatory", "n_pairs", integer=True)),
+    "nPairedAttacksRegistered": (f"{POWER} H1.registered.pairs", power("H1", "registered", "pairs", integer=True)),
+    "hOneMDELow": ("power_prereg.h1_block(h1.json confirmatory.n_pairs)[mde_pid0.2] (realized; DEV-6)",
+                   h1_realized_mde(0.2)),
+    "hOneMDEHigh": ("power_prereg.h1_block(h1.json confirmatory.n_pairs)[mde_pid0.4] (realized; DEV-6)",
+                    h1_realized_mde(0.4)),
+    "hOneMDELowRegistered": (f"{POWER} H1.registered.mde_pid0.2", power("H1", "registered", "mde_pid0.2")),
+    "hOneMDEHighRegistered": (f"{POWER} H1.registered.mde_pid0.4", power("H1", "registered", "mde_pid0.4")),
     "hTwoMDELow": (f"{POWER} H2.registered_W_clean.mde_between_sd0.0_units",
                    power("H2", "registered_W_clean", "mde_between_sd0.0_units")),
     "hTwoMDEHigh": (f"{POWER} H2.registered_W_clean.mde_between_sd0.05_focal_d0",
@@ -805,6 +902,8 @@
                          res("h2", "per_implementation", "VERB", "spearman_ar", "rho")),
     "hTwoSpearmanNG": (f"{ANALYSIS}/h2.json per_implementation.NG.spearman_ar.rho",
                        res("h2", "per_implementation", "NG", "spearman_ar", "rho")),
+    "hTwoSpearmanEmuPooled": (f"{ANALYSIS}/h2.json per_implementation.emu.spearman_ar_pooled.rho",
+                              res("h2", "per_implementation", "emu", "spearman_ar_pooled", "rho")),
     "hTwoSpearmanPreOpen": (f"{ANALYSIS}/h2.json controls.pre_open_only.mean",
                             res("h2", "controls", "pre_open_only", "mean")),
     "hTwoSpearmanRelease": (f"{ANALYSIS}/h2.json sensitivities.W_release.mean",
@@ -822,6 +921,11 @@
                              res("h3", "coverage_placebo", "delta_ic_over_placebo", "mean")),
     "poolOracleShare": (f"{ANALYSIS}/pd2.json oracle_gap_share.primary.share",
                         res("pd2", "oracle_gap_share", "primary", "share", decimals=0, scale=100)),
+    "poolLogLossDiff": (f"{ANALYSIS}/pd2.json paired.primary_vs_V-EW-log.delta_log_loss_b_minus_a.mean",
+                        res("pd2", "paired", "primary_vs_V-EW-log", "delta_log_loss_b_minus_a", "mean", decimals=5)),
+    "poolLogLossDiffCI": (f"{ANALYSIS}/pd2.json paired.primary_vs_V-EW-log.delta_log_loss_b_minus_a.ci",
+                          res_ci("pd2", "paired", "primary_vs_V-EW-log", "delta_log_loss_b_minus_a", "ci",
+                                 decimals=5)),
     "costPerStockWeek": (f"{ANALYSIS}/pd0.json typed.ledger.usd_per_stock_week (x 10^-3)",
                          res("pd0", "typed", "ledger", "usd_per_stock_week", decimals=2, scale=1e3)),
     "costPerRebalance": (f"{ANALYSIS}/pd0.json typed.ledger.usd_per_week",
@@ -912,12 +1016,25 @@
 }
 
 
-SENTENCE_WIRING = {"hOneSentence": (f"{ANALYSIS}/confirmatory.json claims.H1", claim("H1")),
+SENTENCE_WIRING = {"poolSentence": (f"{ANALYSIS}/pd2.json claim_composition_adds", flag("pd2", "claim_composition_adds")),
+                   "hOneSentence": (f"{ANALYSIS}/confirmatory.json claims.H1", claim("H1")),
                    "hTwoSentence": (f"{ANALYSIS}/confirmatory.json claims.H2", claim("H2")),
                    "hThreeSentence": (f"{ANALYSIS}/confirmatory.json claims.H3", claim("H3")),
                    "boundSentence": (f"{ANALYSIS}/frontier.json value_of_text.ci_excludes_zero", bound_outcome)}
 
 
+# DEV-9: when the reversed (descriptive) wording replaces a fallback (see Sentence.reversed)
+REVERSAL_WIRING = {
+    "hOneSentence": (f"{ANALYSIS}/h1.json confirmatory.ci lies above 0 (claim: typed - holistic < 0)",
+                     ci_opposite("h1", ("confirmatory", "ci"), "below")),
+    "hOneAbstract": (f"{ANALYSIS}/h1.json confirmatory.ci lies above 0 (as hOneSentence)",
+                     ci_opposite("h1", ("confirmatory", "ci"), "below")),
+    "poolSentence": (f"{ANALYSIS}/pd2.json paired.primary_vs_V-EW-log.delta_log_loss_b_minus_a.ci lies below 0 "
+                     "(claim: above 0)",
+                     ci_opposite("pd2", ("paired", "primary_vs_V-EW-log", "delta_log_loss_b_minus_a", "ci"), "above")),
+}
+
+
 def _wire() -> None:
     known = {n.name: n for g in GROUPS for n in g.items}
     for name, (source, getter) in WIRING.items():
@@ -929,6 +1046,10 @@
         s = sents[name]
         if s.outcome is None:
             s.outcome, s.source = outcome, source
+    for name, (_source, rev) in REVERSAL_WIRING.items():
+        s = sents[name]
+        if s.reversal is None:
+            s.reversal = rev
     for name, (which, clauses) in CLAUSE_WIRING.items():
         s = sents[name]
         target = s.confirmed_clauses if which == "confirmed" else s.fallback_clauses
@@ -1032,6 +1153,8 @@
         for c in s.confirmed_clauses:
             lines.append(f"%   + only if {c.source}: {c.text}")
         lines.append(f"% if null ({s.ref}): {s.fallback}")
+        if s.reversed:
+            lines.append(f"% if null and the CI lies on the side opposite to the claim (DEV-9): {s.reversed}")
         for c in s.fallback_clauses:
             lines.append(f"%   + only if {c.source}: {c.text}")
         emit(s.name, value, f"{s.desc} | source: {s.source}")
```
</details>

---

## DEV-10 (2026-09-29): a new driver, `experiments/run_layer_b_readers.py`, registers the V-reader-H, -H-spot and -H-guard trials

**Where.** Prereg §4.4 (the registered trial set, M = 73: "on the registered window every name must be present except
the declared not-run V-reader-HS/sel_ew and V-reader-HS/sized") and §5 (V-reader-{H, H-guard, H-spot} "covered on E_t
only (P-1)"); PD1 (§4.1). Audit A3 ("V-reader-H and V-reader-H-spot trial rows, which block PD1, the DSR and Table 4").

**What happened.** `pd1_portfolio.py` exited by its registered strict trial rule: V-reader-H, V-reader-H-spot and
V-reader-H-guard (× sel_ew, sized) were missing from `results/trials.jsonl`. Cause: by the registered deviation P-1
these readers' clean runs are separate generations on E_t (`results/judgments/{h,hspot,hguard}_emu_wclean_et/`,
written by `rts.py clean-readers`), so `run_layer_b.py --variants all` over MAIN's generation skipped them ("no
`*_outlook` feature"; postunlock_core §2.2), and no registered entry point writes their trial rows. `run_layer_b.py`
cannot stack a reader generation, and re-running it with only these variants would re-register and overwrite the
primary run's directory. The hashed `redteam.stages.reader_log` (used by rts.py for H1) and `frontier.py --reader-gens`
(DEV-5) replay the same variants on stacked inputs, but with `register=False`.

**Fix (new file; no hashed file changed).** A thin driver does for these three variants what `run_layer_b.py` does for
every other configuration variant: the primary configuration exactly as `run_layer_b.py` builds it (with the logged
`coverage` waiver, DEV-1: cfg `4ea23931…`), `load_inputs` on W_clean, the reader generation stacked in with
`redteam.stages.stack_raw` (hashed, as rts.py and DEV-5), then
`evaluate(variant_config(cfg, "V-reader-<reader>"), "W_clean", inputs, register=True, write=True)`. The stacked
inputs have their own inputs hash, so each variant gets a new run directory and no existing directory is touched.

**Runs.**
- V-reader-H: reader generation `results/judgments/h_emu_wclean_et` (features SHA-256 `1a762486a612e979…`); cfg `0b36d50421b923af…`, inputs `68a06424e2b1`; run directory `results/eval/0b36d50421b923af/W_clean__68a06424e2b1/`; weekly IC -0.0234 over 113 learned weeks; two trial rows (`V-reader-H/sel_ew`, `V-reader-H/sized`). Coverage flags waived by the logged `coverage` waiver: h_outlook@2024-07-05, h_outlook@2024-07-12 (the two cold weeks, where E_t and so the reader feature do not exist).
- V-reader-H-spot: reader generation `results/judgments/hspot_emu_wclean_et` (features SHA-256 `506787f00257cc85…`); cfg `d9c3775557c0878b…`, inputs `bc88d67f7046`; run directory `results/eval/d9c3775557c0878b/W_clean__bc88d67f7046/`; weekly IC -0.0235 over 113 learned weeks; two trial rows (`V-reader-H-spot/sel_ew`, `V-reader-H-spot/sized`). Coverage flags waived by the logged `coverage` waiver: hspot_outlook@2024-07-05, hspot_outlook@2024-07-12 (the two cold weeks, where E_t and so the reader feature do not exist).
- V-reader-H-guard: reader generation `results/judgments/hguard_emu_wclean_et` (features SHA-256 `359ed92833221e33…`); cfg `97d3df3a23ec85b5…`, inputs `1b5d9ed1a437`; run directory `results/eval/97d3df3a23ec85b5/W_clean__1b5d9ed1a437/`; weekly IC -0.0233 over 113 learned weeks; two trial rows (`V-reader-H-guard/sel_ew`, `V-reader-H-guard/sized`). Coverage flags waived by the logged `coverage` waiver: hguard_outlook@2024-07-05, hguard_outlook@2024-07-12 (the two cold weeks, where E_t and so the reader feature do not exist).
- Calls: `run_layer_b_readers.py --allow-missing coverage --reader H=results/judgments/h_emu_wclean_et --reader H-spot=results/judgments/hspot_emu_wclean_et` (2026-09-29, before H-guard existed), then `--reader H-guard=results/judgments/hguard_emu_wclean_et` after DEV-8. Summary: `results/eval/reader_variants_W_clean.json`. `results/trials.jsonl`: 130 → 136 rows.

**Checks.** Each variant's weekly IC equals the clean IC that `frontier.py` (DEV-5, `clean_by_reader`) computed for the
same reader on the same stacked inputs (H −0.0234, H-spot −0.0235; H-guard −0.0233, recomputed by the re-run of `frontier.py` with the H-guard generation stacked). No decision code changed; the
variant configurations are the registered `pipeline.variants` ones.

**Effect.** The registered trial set is complete except the declared not-run V-reader-HS/* (HS is cut), so PD1 runs
under its registered strict rule, with raw M = 73 as registered.

**Hashes.** New file `experiments/run_layer_b_readers.py`: `6afce3fdffd40d2ccc61b428b8baff33897920ad4d43a9094da2c822cc374a11`. It is not in `CODE_HASHES.json` (written
after the commit). No hashed file changed under this entry.

---

## DEV-11 (2026-09-29): Table 2 gets the R-vote row; Table 3 separates the weekly-mean and pooled Spearman rows

**Where.** Prereg §4.1 (Table 2: per reader in the same slot, band and cap), §3.1 C1-secondary (R-vote is one of the
seven registered contrasts), §3.2 per implementation (emu's confirmatory statistic is the weekly mean; NG and VERB
report the pooled Spearman). Audit findings A9 and A6.

**Changes (presentation only; no statistic is computed differently).**
- `experiments/report/table2_typed_interface.py`: the hashed `READERS` list had no R-vote row, although R-vote's
  registered contrast and descriptives exist (`h1.json`, `frontier.json`, `localization.json`). One row is added,
  ("Typed, majority vote", "R-vote", "R-vote", "typed", True), read by the same code as the R-med row.
- `experiments/report/table3_validity.py`: the first row of panel A put emu's weekly mean (0.119) next to VERB's and
  NG's pooled values. Panel A now has two groups of rows: the weekly mean with its NW CI (emu; items before 09:30 ET;
  placebo; W_release) and the pooled Spearman with its week-block CI (emu on its full frame from
  `h2.json per_implementation.emu.spearman_ar_pooled`, VERB, NG; the pooled placebo for VERB and NG). The units row
  prints thousands separators for VERB and NG as for emu.

**Hashes.**

| File | Before | After |
|---|---|---|
| `experiments/report/table2_typed_interface.py` | `106092b85ffc9514f8e2c04bf4eae2d19b9567b58d2c6d7d69cee6a1fc270ea9` | `0cf14aad42a7f50267a2e5011a04492b7a49b03e4a18769bda2d379f60a41508` |
| `experiments/report/table3_validity.py` | `e50d03a0f5930fd02a9310941a3eb9edf70a5ead5159b333e5c3d00056973065` | `a1bbf0a058f9f4da80053f5c73391fe3135b16a6eade3db7952edb956f378ceb` |

<details><summary>diff</summary>

```diff
--- a/experiments/report/table2_typed_interface.py
+++ b/experiments/report/table2_typed_interface.py
@@ -7,7 +7,7 @@
     .venv/bin/python experiments/report/table2_typed_interface.py --in-dir results/analysis/W_pre_slice --out-dir X
 
 Rows (registered readers; "---" = not run, "n/a" = not applicable): typed programs (ours); typed, no defenses; holistic;
-holistic + guards; holistic + marking; typed, median aggregation; dictionary (LM, standing in for the cut headline-tone
+holistic + guards; holistic + marking; typed, median aggregation; typed, majority vote (DEV-11); dictionary (LM, standing in for the cut headline-tone
 reader); TA-style pipeline and Mono-CoT (S30 names, dagger); code-only.
 Columns: clean selection return of the signal book at the operating point (bps per week; frontier.json
 `clean_by_reader`); static in-band ASR pooled over A1-A7 with its 95% CI clustered on the attack week (h1.json
@@ -40,6 +40,8 @@
            ("Holistic + guards", "H-guard", "H-guard", "H-guard", False),
            ("Holistic + marking", "H-spot", "H-spot", "H-spot", False),
            ("Typed, median aggregation", "R-med", "R-med", "typed", True),
+           # DEV-11 (audit A9): R-vote is a registered C1-secondary reader (prereg §3.1); its row was missing
+           ("Typed, majority vote", "R-vote", "R-vote", "typed", True),
            ("Dictionary (LM)", "LM", "LM", None, False),
            ("TA-style pipeline$^\\dagger$", "TA-lite", None, "TA-lite", False),
            ("Mono-CoT$^\\dagger$", "Mono-CoT", None, "Mono-CoT", False),
--- a/experiments/report/table3_validity.py
+++ b/experiments/report/table3_validity.py
@@ -6,9 +6,10 @@
     .venv/bin/python experiments/report/table3_validity.py         # results/analysis/{h2,h3,pd2,pd3,confirmatory}.json
     .venv/bin/python experiments/report/table3_validity.py --in-dir results/analysis/W_pre_slice --out-dir X
 
-Panel A (columns emu | VERB | NG; a column whose implementation did not run shows "---"): Spearman of E[effect] with
-AR[0,+1] (emu: the confirmatory mean weekly Spearman with its NW 95% CI; VERB and NG: pooled on their samples with
-week-block CIs, descriptive); items created before 09:30 ET; placebo AR[-5,-1]; AUC of P(event type) against 8-K
+Panel A (columns emu | VERB | NG; a cell without its statistic shows "---"): Spearman of E[effect] with AR[0,+1] in
+two groups of rows (DEV-11): the weekly mean with its NW 95% CI (emu's confirmatory statistic, with items created
+before 09:30 ET, placebo AR[-5,-1] and W_release), then the pooled Spearman with week-block CIs (emu on its full
+frame, VERB and NG on their samples, descriptive, with the pooled placebo); AUC of P(event type) against 8-K
 Items 2.02, 1.01/2.01 and 5.02; AUC of vs_expectations against the sign of AR; W_release; units.
 Panel B: H3 dIC with its CI and the Holm-adjusted p (confirmatory.json), on W_clean and W_release; the split into
 trusted typed judgments and untrusted text; the increment over the coverage placebo; composition: IC of the learned
@@ -47,14 +48,17 @@
     def cells(emu: str, other) -> list[str]:
         return [emu] + [other(impl[k]) if ok[k] else L.NOT_RUN for k in IMPLS[1:]]
 
-    rows = [L.row("Spearman, AR$[0,+1]$", *cells(wk(L.get(h2, "confirmatory")),
-                                                 lambda x: pooled(x.get("spearman_ar")))),
-            L.row("\\quad items before 09:30 ET", *cells(wk(L.get(h2, "controls", "pre_open_only")),
-                                                         lambda x: L.NOT_RUN)),
-            L.row("\\quad placebo AR$[-5,-1]$", *cells(wk(L.get(h2, "placebo_ar_m5_m1")),
-                                                       lambda x: pooled(x.get("spearman_ar_placebo")))),
-            L.row("\\quad $W_{\\mathrm{release}}$", *cells(wk(L.get(h2, "sensitivities", "W_release")),
-                                                           lambda x: L.NOT_RUN))]
+    # DEV-11 (audit A6): emu's confirmatory statistic is the WEEKLY mean (NW CI); VERB and NG report the POOLED
+    # Spearman (week-block CI). Each statistic has its own group of rows, so that no row compares different statistics.
+    rows = [L.row("Spearman, AR$[0,+1]$, weekly mean", wk(L.get(h2, "confirmatory")), L.NOT_RUN, L.NOT_RUN),
+            L.row("\\quad items before 09:30 ET", wk(L.get(h2, "controls", "pre_open_only")), L.NOT_RUN, L.NOT_RUN),
+            L.row("\\quad placebo AR$[-5,-1]$", wk(L.get(h2, "placebo_ar_m5_m1")), L.NOT_RUN, L.NOT_RUN),
+            L.row("\\quad $W_{\\mathrm{release}}$", wk(L.get(h2, "sensitivities", "W_release")), L.NOT_RUN,
+                  L.NOT_RUN),
+            L.row("Spearman, AR$[0,+1]$, pooled", pooled(L.get(h2, "per_implementation", "emu", "spearman_ar_pooled")),
+                  *[pooled(impl[k].get("spearman_ar")) if ok[k] else L.NOT_RUN for k in IMPLS[1:]]),
+            L.row("\\quad placebo AR$[-5,-1]$", L.NOT_RUN,
+                  *[pooled(impl[k].get("spearman_ar_placebo")) if ok[k] else L.NOT_RUN for k in IMPLS[1:]])]
     for key, lab in SEC_ROWS:
         rows.append(L.row(lab, *cells(L.num(L.get(h2, "sec_8k", key, "auc"), 3),
                                       lambda x, k=key: L.num(L.get(x, "sec_8k", k, "auc"), 3))))
@@ -64,7 +68,8 @@
     n_emu = L.get(h2, "confirmatory", "n_units")
     rows.append(L.row("Units (weeks)", f"{n_emu:,}".replace(",", "{,}") + f" ({L.get(h2, 'confirmatory', 'n_weeks')})"
                       if n_emu else L.NOT_RUN,
-                      *[f"{impl[k].get('n_units')}" if ok[k] else L.NOT_RUN for k in IMPLS[1:]]))
+                      *[f"{impl[k].get('n_units'):,}".replace(",", "{,}") if ok[k] else L.NOT_RUN
+                        for k in IMPLS[1:]]))
     return rows
 
 
```
</details>

---

## DEV-12 (2026-09-29): a presentation script, `experiments/report/make_numbers_extra.py`, for registered-output macros

**Where.** Prereg §3.4 (sentences and numbers generated by script); positioning_v3 D-8. Appended now; the script has
existed since 2026-09-29 (outline_v2 §5.2) and its docstring carried this entry's text.

**Change.** "A new presentation script, experiments/report/make_numbers_extra.py, writes
paper/generated/numbers_extra.tex: registered-output macros that the validity-first rewrite needs and make_numbers.py does
not define (outline_v2 §5.2). It reads registered outputs only, through make_numbers.registered_result, plus the stated
records (power_prereg.json, prereg_v1.yaml, this file, results/pre/d3_pre.json, FEED-1, VERB's MANIFEST); it edits no
hashed file and computes no new statistic (every value is a field of an existing output, rounded half-up)."

**Effect.** Touches a confirmatory statistic: **no**.

## DEV-13 (2026-09-29): presentation changes after the round-1 paper reviews (descriptive H3 clause, exact intervals, labels)

**Where.** docs/reviews/paper_v2_r1_{pc,stats,editor,ground}.md; resolution in docs/reviews/paper_v2_r1_resolution.md.
Prereg §3.1-§3.4 (result sentences, p format), §4.1 (PD1 table), §9-§10 (red-team reporting). The pre-registration file
is not edited.

**1. H3 descriptive clause (as DEV-9 for H1 and PD2).** The registered H3 fallback ("No increment was detected at MDE
≈ …") is true, but the paper's gloss of it was not; the H3 estimate is −0.0155 with its 95% CI [−0.0294, −0.0016]
wholly below 0 (`h3.json`, `confirmatory.json`). A new generated macro `\hThreeReversed` adds, only when the Holm
decision is the fallback **and** the CI lies wholly below 0, the descriptive clause "The estimate is negative and its
95% interval lies below zero: in this testbed the typed judgments lowered the pooled logit's rank IC. The registered test
was one-sided for an increment, so this is an estimate, not a tested harm." It is printed after the registered sentence,
which stays verbatim. No new test.

**2. Abstract clauses.** `\hThreeAbstractText` and `\hOneAbstractText` replace `\hThreeAbstract` and `\hOneAbstract`
in the abstract (editor review A1): "the confirmatory test detects no rank information over code-only features, and the
estimate is negative" and "reading items one at a time with guard questions did not lower attack success below a
holistic reader's (10.5% vs. 0.1%)", switched by the same registered outputs. The registered sentences are unchanged.

**3. p format.** `\hTwoPholmBound` prints H2's Holm-adjusted p as "< 10^-6" when it is below 10^-6, and
`\hTwoSentenceBound` is the registered H2 sentence (make_numbers.py's template for the realized outcome) with that bound
in place of 3.3 × 10^-20 (stats review minor 4: the NW t-approximation is not reliable that far into the tail). The exact
value stays in `confirmatory.json`.

**4. Exact and Wilson intervals for boundary counts.** Where at most one attacked unit succeeded, the percentile
cluster bootstrap collapses (0.0 [0.0, 0.0]); the paper prints the exact Clopper-Pearson 95% interval instead, clustering
ignored (`\asrZeroUpper` = 2.1% for 0 of 173; Table 4 marks such cells). D2's accuracy among probes that named a firm
gets its Wilson 95% interval (`\dTwoCondNamesCI`). Realized MDEs of H2 and H3 are printed by the registered formula
(z_holm × realized SE; `\hTwoMDERealized` 0.031, `\hThreeMDERealized` 0.021), as DEV-9 did for H1.

**5. Registered values newly printed** (fields of existing registered outputs or records; no new statistic): the frame
flow (`h2.json frame`), H3's IC levels and log losses (`h3.json ic`, `dm_logloss`), M1b's learner attenuation
(`results/pre/mechanical_checks.json`), PD1 selection alphas and the M = 2 DSR (`pd1.json`), MDEs of D1, D2, NG and VERB
and D1's simulated false-move rates (`power_prereg.json`, `simulations/null_rates.json`), the demotion strata and
per-family rows of other readers (`h1.json`), the bound-check split into entry and protection checks (`bound.json`), PD2's
leave-one-program-out range (`pd2.json`, printed only if every interval covers 0), protocol constants (windows as prose
dates, D1 schedule months, family counts), and the date of the literature search (LIT-1 log).

**6. Table and figure presentation.** `table1_validity.py`: two side-by-side panels (emu; VERB), NG's rows moved to an
appendix table (`tableC_ng.tex`), cost rows removed, CIs in \scriptsize. `table3_planted.py`: an explicit H1 contrast row,
a no-defenses column, the demotion row, exact intervals for boundary counts, "> 0.99" for p = 1, the all-dash TA-style
row and the bound rows removed (the bound is reported in App. A). `table4_horizon_ops.py`: plain book labels, IC levels,
the selection-alpha column, no cost panel. `table4_portfolio.py` (hashed before the commit): plain book labels, one line
per book with the alpha interval, \scriptsize intervals. `table_models.py`: row labels and column widths.
`fig1_validity_architecture.py`, `fig2_validity.py`, `fig_d_lookahead.py` (panel a marks the registered permutation
decision instead of drawing the descriptive bootstrap intervals, which prereg §7.2 found too narrow), `fig_f_horizon.py`:
type sizes, labels and encodings. `make_lit_numbers.py`: one literature count (`\litTangPapers`). `scripts/check_numbers.py`:
a macro counts as a placeholder only by its last definition in main.tex's input order.

**Hashes (after; the files not listed as hashed were written after the commit).**

| File | Before | After |
|---|---|---|
| `experiments/report/table4_portfolio.py` | `9b50bb6b83991f43f194734e81b26d8958f16c28c074477cd6fc45be712841e7` | `0c1aa00fb8ec1df6257cb4015148aa47e0f4bf5dfd809253e716a23d06023c6f` |
| `experiments/report/make_numbers_extra.py` | (DEV-12) | `16a343b2e9b2875dd73ffd226e445a714d99c5fbde1b1f2c2565587ff7521e8c` |
| `experiments/report/make_lit_numbers.py` | — | `19fd32dd87a76c759a99aec530432687b62b59d1b3cf3336cd1ea7f0cc4fa32a` |
| `experiments/report/table1_validity.py` | — | `4979a26906a579f1028507f3e9964786799daf2bef01e0b3a4d11237d8efa278` |
| `experiments/report/table3_planted.py` | — | `c91c613b32e68e24ea323eec2a9cc864f7528a903ec7fadd9f3ed6602ed419f2` |
| `experiments/report/table4_horizon_ops.py` | — | `f8062e579e3b34811ed0ef212546f624e1f8fd2b1797690f07325fbf543187d2` |
| `experiments/report/table_models.py` | — | `8a64a6f3feb6ec17e466b4bf989c004126028f89f8141514aa8427100d6e8c99` |
| `experiments/report/fig1_validity_architecture.py` | — | `55c4d8a9dced9e2a1de0709d3fca46361e194b44be5c4c36d79e9e2b8c3f9db4` |
| `experiments/report/fig2_validity.py` | — | `7e2e5fd6e9a6e733357338101b68736e148f9fdddefe4ac947e251c626fe792d` |
| `experiments/report/fig_d_lookahead.py` | — | `518deb0c7470ab4887e2824bff20357ce9fb34cc788b8d50230aff2f7e8e5854` |
| `experiments/report/fig_f_horizon.py` | — | `3deb01909e3e300f1064b73b38467815693f9e8f902c39f327ba83529b6aa346` |
| `scripts/check_numbers.py` | — | `5fdc9e7ae6b8de71c0282c5aeb0687eac6dac0946742c26265662e063651841b` |

**Effect.** No registered number, test, Holm decision or clause condition changes. Touches a confirmatory statistic:
**no**.

## DEV-14 (2026-09-29): presentation changes after the round-2 paper reviews (layout, labels, generated Jev-slot wording)

**Where.** docs/reviews/paper_v2_r2_{ac,devil,figs}.md; resolution in docs/reviews/paper_v2_r2_resolution.md. Prereg
§3.4 (sentences and numbers generated by script), §4.1 (PD1 table), §9-§10 (red-team reporting). The pre-registration
file is not edited.

**1. Table and figure presentation (no new statistic).** `table1_validity.py`: estimates and 95% intervals in separate
columns at one type size, top-aligned panels, positives in the row labels, ρ and κ in two rows, `\ref` in the note; with
a Jev row, the panels stack (figure review J-1); Table C (NG) at natural width. `table4_horizon_ops.py`: two tabulars
with their own headers and a leading status column; the DSR moved to App. E. `table3_planted.py`: two panels with their
own headers; no blank cells ("---", "ref.", "n/a"); exact-interval marks hung right; the H1 contrast on the holistic
reader's row (status C); the adaptive attacker in the note. `table4_portfolio.py` (hashed before the commit): the note in
a minipage, the script named in the header, −∞ defined. `table_models.py`: pin breaks only before the snapshot date,
"Jev (not run)", shorter headers. `fig1_validity_architecture.py`: the trusted path a double line drawn under the
interface bar, labels off its path, 1.0 pt program borders, numbers outside math, 2.86 in tall. `fig2_validity.py`: open
markers only for placebo windows; sensitivities filled light indigo; window named in each group header; the E8 rows
(reference readers from publication, paired emu − FinBERT, FinBERT placebo); "after a close or outside a session"; saved
at the exact text width. `fig_f_horizon.py`: a schematic timeline panel (no data), minor ticks, exact column width.
`fig_d_lookahead.py`: development slice as grey squares, a 0-100% axis, exact text width. `_validity.py`: distinct
marker shapes for Jev (hexagon) and the reference readers. `make_lit_numbers.py`: three lopezlira2026chatgpt numbers
(drift Sharpe 6.54 → 1.22; unprofitable at 20 bps), checked against the summary.

**2. Generated Jev-slot wording.** `make_numbers_extra.py` adds `\jevPanelStatus`, `\jevPanelTiming` and
`\jevRecallClause` (empty without a Jev row), so the slot's timing phrase is computed from where the Jev rows' weeks lie
relative to Jev's release (configs/system_v1.yaml windows; docs/jev/wikipedia_jev.txt), instead of the hard-coded "all
before Jev's release" (devil's-advocate review m4); "descriptive" is replaced by "registered estimate" (m5).

**3. Checks.** `scripts/check_jev_slot.py` (new; writes nothing under paper/ or results/) renders the Jev branch with a
mock Jev row (a copy of VERB's row) in a temporary copy of paper/ and fails on overfull boxes or more than 12 pages.

**Hashes (before = DEV-13's after, or the file as of the round-1 resolution).**

| File | Before | After |
|---|---|---|
| `experiments/report/table4_portfolio.py` | `0c1aa00fb8ec1df6257cb4015148aa47e0f4bf5dfd809253e716a23d06023c6f` | `3629bd127f78b4f50e7994057a12729b3493a115f82c3541015b3fe11f8d994f` |
| `experiments/report/make_numbers_extra.py` | `16a343b2e9b2875dd73ffd226e445a714d99c5fbde1b1f2c2565587ff7521e8c` | `9159b23d13de19267c82f8018e62ac1aec7f17a4038c3609652d2ef8700fedb7` |
| `experiments/report/make_lit_numbers.py` | `19fd32dd87a76c759a99aec530432687b62b59d1b3cf3336cd1ea7f0cc4fa32a` | `173e293b8630463fb1c492869d461d98025da6cb4e896f05a5a165f52bd74c11` |
| `experiments/report/table1_validity.py` | `4979a26906a579f1028507f3e9964786799daf2bef01e0b3a4d11237d8efa278` | `d51592c9bab1690acf487344726abac16f7ed5b305b32570d95a108a546eaf0c` |
| `experiments/report/table3_planted.py` | `c91c613b32e68e24ea323eec2a9cc864f7528a903ec7fadd9f3ed6602ed419f2` | `80ad7fd3e34d3b3c4de0c08104111d8e6be235e1fd999146830f584d0a0bfc89` |
| `experiments/report/table4_horizon_ops.py` | `f8062e579e3b34811ed0ef212546f624e1f8fd2b1797690f07325fbf543187d2` | `01a57c5abe382439068176eb5e5861d3e6ae3d3cd12a323a3d96c0418167d23e` |
| `experiments/report/table_models.py` | `8a64a6f3feb6ec17e466b4bf989c004126028f89f8141514aa8427100d6e8c99` | `f6a5e2e84e624008bf75413b952d6ef561278fa50c9f9a528ff0e328067e7fed` |
| `experiments/report/fig1_validity_architecture.py` | `55c4d8a9dced9e2a1de0709d3fca46361e194b44be5c4c36d79e9e2b8c3f9db4` | `40f58fb57934629419d9f50fd73903c574711bcc0c86ec7e69d1921d85f00c57` |
| `experiments/report/fig2_validity.py` | `7e2e5fd6e9a6e733357338101b68736e148f9fdddefe4ac947e251c626fe792d` | `17b983f51af80374029a97f6a8723de84f394dddd58ea9bfe54ec1fb4de836f0` |
| `experiments/report/fig_d_lookahead.py` | `518deb0c7470ab4887e2824bff20357ce9fb34cc788b8d50230aff2f7e8e5854` | `69988b841804b75432f1eb582924ebd3693105e4994425d0c9e52c2176c145e6` |
| `experiments/report/fig_f_horizon.py` | `3deb01909e3e300f1064b73b38467815693f9e8f902c39f327ba83529b6aa346` | `598a6d22e35176923654d7db55e4138e1ead9ff44e4919157016e3aee79078ab` |
| `experiments/report/_validity.py` | `41601fc62453ff6a…` (round-1 state) | `fa8b05c02945ae2abcd0dd336eb5ab5fbe6699a9e87fed06ea29e9d30aea415e` |
| `scripts/check_jev_slot.py` | — (new) | `253486515b937e7490ccc0bb5ca82c13b2465e978c45e7558da9bbf498b035d6` |

**Effect.** No registered number, test, Holm decision or clause condition changes; the registered sentences are printed
verbatim. Touches a confirmatory statistic: **no**.

## Exploratory analyses written after the unlock (2026-09-29; not pre-registered, not deviations)

These files add no registered analysis and change no hashed file. Every output is labelled "exploratory (not
pre-registered)" in its JSON `meta` (`label`, `exploratory: true`, `why_exploratory`) and payload, in its table
notes, and in every macro (names prefixed `\ex`, comments carrying the label). Outputs go to `results/exploratory/`
and `paper/generated/exploratory_numbers.tex`; `make_numbers.py` never reads them. Nothing here tunes the system: no
configuration, threshold or variant was chosen from them.

| File (SHA-256) | Output | Content |
|---|---|---|
| `experiments/exploratory/_xcommon.py` (`9b64473f9bfcf1ed…`) | — | shared writer (label, meta) |
| `experiments/exploratory/pd3_stores_off.py` (`3d22840ab4bb93df…`) | `results/exploratory/pd3_stores_off.json` | PD3's registered estimand (§4.1) on a stores-off Layer-B replay over the MEM generations, built after the unlock: M-epi off (MEM arm 1's N23 features), M-work off (MEM arm 2's T features), both off; S30; ΔIC, Δ log loss, ΔIR with intervals. Registered PD3 stays "not run" (DEV-7) |
| `experiments/exploratory/pd1_extended.py` (`8bd5a05b461bb689…`) | `results/exploratory/pd1_extended.json` | PD1 extension: reversal, low-volatility, B8 logistic, LM-dictionary reader books; CAGR, Calmar, cost drag, IR vs EW-U100 [CI], FF5+Mom attribution; PE-6 stress table (the two named 2025 tariff episodes; 17 mechanical weeks by date only); TA-lite / Mono-CoT books and weekly IC on their S30 × 60-week stock-weeks vs the typed system on the same stock-weeks. Registered DSR / Random-20 values are only repeated from `pd1.json` |
| `experiments/exploratory/make_exploratory.py` (`3e92b95e3e9076ef…`) | `paper/generated/exploratory_numbers.tex`, `results/exploratory/tables/tableX{1,2}_*.tex` | `\ex…` macros for the paper's exploratory appendix; the extended tables are repository tables (PE-6, PE-16), not input by the paper |
| `experiments/exploratory/e1_asr_mechanism.py` (`de31f47bfe8860f9…`) | `results/exploratory/e1_asr_mechanism.json` | E1: why typed ASR > holistic. Learned untrusted slopes, clean text influence (sd of the capped L^W on E_t, text-dependent book share), ASR per unit influence, the attack's shift of the attacked name, influence-matched and sign-flipped replays of the RT-S units, per-family ASR (incl. nodef), per-feature contributions and knock-outs of the typed successes. $0 replays; the as-logged replay reproduces the registered score tables exactly (7,833 rows, 0 mismatches) |
| `experiments/exploratory/e2_priced_horizon.py` (`a65a234031615039…`) | `results/exploratory/e2_priced_horizon.json`, `results/exploratory/figdata/e2_*.csv` | E2: horizon profile of item E[effect] (event and decision time, AR windows, overnight gap, traded holding week) and of N3 f_net at the stock-week level; timing shares relative to the fill. The rebuilt H2 frame reproduces h2.json's statistic exactly |
| `experiments/exploratory/e3_lookahead_by_month.py` (`27993bb543b3358f…`) | `results/exploratory/e3_lookahead_by_month.json`, `results/exploratory/figdata/e3_*.csv` | E3: D1 AUC per month with name-permutation null bands (AUC and p re-derived equal d1.json and W_pre_slice/d1.json), single months, H2/H3 before vs after the snapshot release (registered W_release values repeated) and by D1-flagged months |
| `experiments/exploratory/e4_trader_attribution.py` (`17b29852462774f0…`) | `results/exploratory/e4_trader_attribution.json`, `results/exploratory/figdata/e4_weekly_layers.csv` | E4: D-sized vs EW-U100 split into trusted selection (V-J0 D-sel-EW), sizing/risk overlay and text; turnover, cost drag, text entrants vs displaced names. Books re-run from registered weights reproduce the logged returns exactly |
| `experiments/exploratory/fig_exploratory.py` (`fd5040f7897c0e65…`) | `results/exploratory/figures/figX_{e2_horizon,e3_lookahead}.{pdf,png}` | preview figures for E2 and E3 from the figdata tables (labelled inside the figure); not written to paper/figures/ |
| `experiments/exploratory/make_exploratory.py` (2026-09-29 update, `3e92b95e3e9076ef…` → `f75ffc543616c240…`) | `paper/generated/exploratory_numbers.tex` | adds the `\exE…` macros of E1–E4 (86 new macros; 126 in total); every earlier macro line is unchanged (diff: additions only) |
| `experiments/exploratory/e6_publication_timing.py` (`514d29bd8284d886…`) | `results/exploratory/e6_publication_timing.json`; minute bars cached in `data/raw/bars/minute_sip/` | E6 (round-1 PC review B1): the registered H2 statistic by publication time; a publication-aligned criterion from vendor minute bars (first trade after publication to the close of d0+1, market-adjusted), overall, by timing, by event type, on W_release; W_release without price reports; paired H2 − placebo; partial Spearman given AR[−5,−1]; peakedness of emu's effect distributions; a linear approximation of the residual after the price-report classifier's misses. The frame reproduces h2.json exactly |
| `experiments/exploratory/e7_reference_readers.py` (`6a5b805c8d6d6fe2…`), `e7_finbert_worker.py` (`820bfbade369510a…`) | `results/exploratory/e7_reference_readers.json`, `results/exploratory/e7_finbert_scores.csv` (no text) | E7 (round-1 PC M3, stats B3): FinBERT tone (yiyanghkust/finbert-tone @ 4921590d, local CPU, $0) on every frame text vs AR[0,+1] and its placebo, item AUC; the paired week-block bootstrap of VERB − emu on VERB's registered units (reproduces the registered pooled rows) |
| `experiments/exploratory/r1_make_numbers.py` (`860e3dfe7d70f318…`) | `paper/generated/exploratory_numbers_r1.tex` | `\ex` macros of E6, E7 and the Holm multiplicity note over the registered D1 windows |
| `experiments/exploratory/e5_table_concepts.py` (update, `aa47fcfa50106f17…`) | `paper/tables/table2_concepts.tex` | adds E6's publication-aligned column; types below 5% of the frame omitted; italics mark intervals covering 0 |
| `experiments/exploratory/e8_timing_robustness.py` (`720fa0a1ae3fea55…`) | `results/exploratory/e8_timing_robustness.json` | E8 (round-2 AC review W3, W8; devil's-advocate B1, M5, M6, m1): on the registered H2 frame (reproduced to 1e-12 through E6), emu, FinBERT and Loughran-McDonald tone against AR[0,+1], the placebo and E6's publication-aligned criterion, with paired weekly emu − FinBERT and emu − dictionary differences; the SEC clock (earnings items vs the nearest Item 2.02 acceptance); edits after creation; the after-close-or-non-session group's composition; byline classes (FEED-1); frame sensitivities (all pairs with an effect answer; kept pairs of every type). $0, local; the minute bars are E6's cache |
| `experiments/exploratory/r2_make_numbers.py` (`460ed532d971dde7…`) | `paper/generated/exploratory_numbers_r2.tex` | `\ex` macros of E8, the E2 row for items whose event session is the decision session, and the count of exploratory analyses |
| `experiments/exploratory/e5_table_concepts.py` (update, `aa47fcfa50106f17…` → `e740bb02e472538a…`) | `paper/tables/table2_concepts.tex` | a dagger marks intervals that exclude 0 (italics dropped); two-row header; `\ref` in the note |
| `experiments/exploratory/e5_box1_examples.py` (update, `c8cbaee6bc8b6a91…` → `d272bd599d255d7c…`) | `paper/tables/box1_examples.tex` | Box 1 as a light framed box (primitive skips, so acmart does not warn); levels named; no note that repeats the caption |
| `experiments/exploratory/e5_refresh.sh` (update, `e5e4acceb66cf656…` → `897146da44d26759…`) | — | also runs r2_make_numbers.py |

## Provenance note (2026-09-29, orchestrator): registration pushed before unlock
GitHub's event log for the private repository records the PushEvent of the pre-registration commit
b6796d65d8ba8761b6cc6c69bfd9d583ac4bf302 at 2026-09-28T09:42:02Z; the lock key (results/locks/clean_unlocked) was
written at 2026-09-28T09:42:22Z and its commit 8e870e97b3fba855c05aa632476cc7c0adc4d46e pushed at 2026-09-28T09:42:24Z
(`gh api repos/<owner>/<repo>/events`, PushEvent rows). The registration therefore reached the remote 20 s before the
lock opened. The paper says "hash-committed before the evaluation window was unlocked (not deposited with a registry)".
