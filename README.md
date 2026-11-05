# Track or Follow? A Jev-Compatible Multi-Agent Trading System When Web News Lags the Market

Code and result files for the paper. The system's agents share only point-in-time stores and answer typed questions
through the interface of Jev; code sets every position under a proved bound on how far text can move it. The release
holds everything behind the paper's numbers, tables and figures, except the licensed news text.

## Contents

| Path | What |
|---|---|
| `src/jevtrader/` | the multi-agent trading system: point-in-time stores, question programs, reader backends (emu, Jev, NG, verbalized), decision layer with the text-influence bound, red-team harness, evaluation |
| `configs/` | run configurations; `prereg_v1.yaml` records the SHA-256 digests committed before the evaluation window was unlocked |
| `docs/design/preregistration_v1.md` | the hash-committed pre-registration (SHA-256 `b8bf8cf6e7135b963cd2069f83214019be4df6ba9bebabbfd81dd41470984b9a`) |
| `docs/design/prereg_deviations.md` | the deviation log |
| `docs/design/question_bank_v1.{md,json}` | the frozen bank of typed questions |
| `experiments/` | generation runs, registered analyses (`analysis/`), exploratory analyses (`exploratory/`, `v2/`), the Jev panel (`jev_panel/`) and the generators of the paper's numbers, tables and figures (`report/`) |
| `results/` | machine-written results read by the generators; `results/release/` holds the item-level tables |
| `paper/generated/`, `paper/tables/`, `paper/figures/` | the generated macros, tables and figures of the paper |
| `scripts/` | data download and table builders, the item-table export, and the check that the paper types no number by hand |
| `docs/jev/`, `docs/ng/`, `docs/sec/`, `docs/pricing/` | archived public documentation the code reads (model schemas, 8-K items, prices) |
| `literature/summaries/` | summaries of cited papers whose numbers the generators check |

## Data

News comes from Benzinga, delivered through the Alpaca news API (headline and summary); daily bars from Alpaca; filings
and macro series from SEC EDGAR and ALFRED. The news text is licensed and not included. The item-level tables in
`results/release/` key each item unit as `t|symbol|eid`, where `t` is the weekly decision date and `eid` is `"news:"`
followed by the first 16 hex digits of SHA-256 of the Benzinga item id, so holders of the feed can rebuild the frame.

- `h2_frame_emu.parquet`: the registered H2 frame with the confirmatory reader's typed judgments (`E_effect`, top event
  type, event-type and vs-expectations distributions as JSON) and the labels: event session `d0`, `ar` = AR[0,+1],
  `ar_placebo` = AR[-5,-1], and the 8-K concept indicators `sec_*`.
- `readers_*.parquet`: every news answer of the Jev (`jev_*`), verbalized (`verb_*`) and NG (`ng_*`) generations,
  keyed by unit; request keys and state digests are SHA-256 values.

`MANIFEST.json` lists row counts and SHA-256 digests.

## Reproducing

Python 3.12, then `pip install -r requirements.txt` and `export PYTHONPATH=src`.

```
python scripts/export_item_tables.py --check     # H2 confirmatory, placebo, pre-open and post-release rows from the released frame
python experiments/report/make_numbers.py         # paper/generated/numbers.tex
python experiments/report/make_numbers_extra.py   # paper/generated/numbers_extra.tex
python experiments/report/table1_validity.py      # and the other table*/fig* scripts in experiments/report/
```

Six feed-composition macros of `make_numbers_extra.py` (`feed*`) are computed from the licensed news table and checked
against its SHA-256; without the table they are written as placeholders, and their values are in
`data/processed/feed_provenance.json`. Rebuilding the item frame, the states and the labels from scratch needs the news
feed, daily bars and SEC filings (download scripts in `scripts/`, credentials in a `.env` file). Model calls need API
credentials; `configs/backends.yaml` holds the run budgets.

Brokerage order execution for forward paper trading is not part of the paper and is not included.
