"""Point-in-time fundamentals from SEC XBRL companyfacts (design_v1 sections 2.4 and 5.4; V-B1, V-M14).

Every XBRL fact carries its fiscal period (start/end), the form that reported it, its accession number and the date
it was `filed`. A fact becomes usable on the first session strictly after its filing date (filings accepted after the
close would otherwise leak into that day's decision). Later filings may restate a period; as of date d we use, for
each period, the most recently filed value with filed < d.

Tags of one concept often measure different things (PFE's 10-Ks tag both total `Revenues` and a narrower
contract-revenue line, MA's contract-revenue tag is gross revenue before rebates, AMT's is non-lease revenue), while
other firms alternate tags for the same measure between filings (NVDA's FY2020 10-K tags contract revenue, its 10-Qs
`Revenues`). So a concept's facts are first merged per period: the *primary* tag is the first tag in priority order
among those reporting the latest period (for revenue, a tag that fails the income-statement identity revenue - costs
= operating income in its latest annual filing loses to one that passes: BLK tags a sub-line as `Revenues`); every
period takes the primary's facts, else those of a tag that measures the same thing around that period (>= 90% of
the periods both report within about a year agree within 2%). Tags that measure something else are never used, so a
quarter is never a difference of two measures. Fiscal quarters then come from the merged facts:
- 3-month facts are used directly;
- otherwise quarters come from year-to-date differencing *within one fiscal year* (facts sharing a fiscal-year
  start): Q2 = H1 - Q1, Q3 = 9M - H1, Q4 = FY - 9M (V-M14; 10-Q cash-flow statements are YTD only). Each operand is
  its latest vintage filed before d; the subtrahend is taken from the minuend's own filing when that filing reports
  it (same vintage), so a restatement is never differenced against an unrestated figure when both are available,
  and a minuend re-reported only as a later year's comparative is paired with the subtrahend's vintage (RECAST_DAYS);
- a missing fiscal Q4 without a 9M figure falls back to FY minus the three known quarters.
TTM = sum of the last four consecutive quarters. Stock items (assets, debt, equity) use the latest instant value,
preferring the higher-priority tag for the same period.

Levels (V-B1). Market value = RAW close at d x the latest share count filed before d, adjusted for splits between
the share count's reference date and d (`data.prices.SplitHistory`, known from each split's ex-date). Adjusted closes
are never multiplied by share counts, and per-share XBRL EPS is never used: comparative per-share figures are restated
after splits, so TTM sums of EPS mix bases. Earnings yield = TTM net income / market value.

Share counts come from the cover page (dei EntityCommonStockSharesOutstanding, reference date = cover date). Firms
with several share classes report it only per class, which companyfacts drops; for them the weighted-average basic
shares of the latest quarter are used (reference date = filing date, because GAAP restates it for splits effected
before issuance), then the balance-sheet count. XBRL counts are sometimes mis-scaled (CCL's 10-K of 2021-01-26 tags
932,485,510,000 cover shares, 1000x; ZTS 2019-08 0.05x; some MCD/KO/TER weighted averages are in millions), so a
count is used only if it is within SHARES_BAND = [1/2, 3] of another source's count or of its own previous report,
all put on d's split basis (`pick_shares`); staleness is judged at the decision date d. The band is asymmetric
because real counts only jump up (stock mergers: IFF x2.4 after DuPont N&B in 2021-02, FIS x2.0); a fall below half
needs a reverse split, which the split factor already applies (WFC's 10-Q of 2023-08-01 tags 1.82B against 3.75B).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# concept -> candidate us-gaap tags in priority order (per company, see the module doc for how one is chosen).
# Revenue: the net-revenue line of financials first (AXP, GS, JPM, MS, WFC report it as their total); then total
# `Revenues`, which is the income-statement total wherever a narrower ASC 606 line is also tagged (PFE, AMT, CEG, MA,
# COF, CHTR, GM, energy firms whose total includes derivative gains). Gross interest income is not a revenue measure
# (it tracks rates: GS "revenue" 81B in 2024 against about 50B of net revenue), so a bank without a net-revenue tag
# has no revenue (TFC).
FLOW = {
    "revenue": ["RevenuesNetOfInterestExpense", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
# Flows loaded only to check which revenue tag is the income-statement total (never features).
IDENTITY = {"operating_expenses": ["OperatingExpenses"], "costs_and_expenses": ["CostsAndExpenses"]}
SERIES_AGREE_TOL = 0.02  # two values of one period "agree" within 2%
SAME_MEASURE_SHARE = 0.9  # tags measure the same thing if >= 90% of their (nearby) common periods agree
NEAR_DAYS = 400  # "nearby" common periods: ending within about a fiscal year of the period to fill ...
NEAR_MIN = 3  # ... or the 3 nearest common periods if none is that near
IDENTITY_TOL = 0.01  # |revenue - costs - operating income| <= 1% of revenue
RECAST_DAYS = 300  # a flow value filed later than this after its period end is a comparative in a later year's report
STOCK = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}
# share-count sources in order of preference: concept, (taxonomy, tag), split reference ("end" or "filed")
SHARE_SOURCES = [
    ("shares_cover", ("dei", "EntityCommonStockSharesOutstanding"), "end"),
    ("shares_wavg", ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"), "filed"),
    ("shares_bs", ("us-gaap", "CommonStockSharesOutstanding"), "filed"),
]
SHARES_MAX_AGE_DAYS = 400  # a share count older than this (as of d) is not used
SHARES_BAND = (0.5, 3.0)  # plausibility band of a count against its references, and of the 1-year share change + 1
# Companies whose share counts exist only per share class (companyfacts drops dimensioned facts). Per-class facts
# come from each filing's XBRL instance (scripts/download_class_shares.py -> data/raw/sec/class_shares/<SYM>.json)
# and are combined into one count on the listed line's price basis:
# - BRK.B: cover-page Class A x 1,500 + Class B ("Each share of Class A common stock is convertible ... into 1,500
#   shares of Class B common stock", 10-K FY2025, sec.gov/Archives/edgar/data/1067983/000119312526083899/
#   brka-20251231.htm);
# - V: diluted weighted-average Class A shares, which Visa reports on an as-converted basis (classes B and C
#   included), i.e. in Class A price terms.
CLASS_SHARES = {
    "BRK.B": ("shares_cover", "EntityCommonStockSharesOutstanding",
              {"CommonClassAMember": 1500.0, "CommonClassBMember": 1.0}),
    "V": ("shares_wavg", "WeightedAverageNumberOfDilutedSharesOutstanding", {"CommonClassAMember": 1.0}),
}
# Cash-flow fields (cfo, capex; FCF yield, accruals, cash-flow margins) enter states and features only when the
# YTD-differencing tests pass (design_v1 section 5.4); tests/test_fundamentals.py covers them.
CASHFLOW_READY = True

_FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F", "10-KT")


@dataclass
class FactTable:
    """Long table: concept, tag, prio, start, end, val, filed, form, accn, days (duration; NaN for instants)."""

    df: pd.DataFrame


def load_facts(path: Path) -> FactTable:
    """Facts for one company. Predecessor-entity files saved next to it as `<SYM>__pred<k>.json` (companies
    that moved to a new holding-company CIK) are merged in. All candidate tags of a concept are kept, each with its
    priority rank (see CompanyState for how tags are combined)."""
    path = Path(path)
    files = [path] + sorted(path.parent.glob(f"{path.stem}__pred*.json"))
    rows = []
    for f in files:
        facts = json.loads(f.read_text()).get("facts", {})

        def take(concept: str, taxonomy: str, tag: str, prio: int) -> None:
            node = facts.get(taxonomy, {}).get(tag)
            if not node:
                return
            for unit, items in node["units"].items():
                if unit not in ("USD", "USD/shares", "shares"):
                    continue
                for it in items:
                    if it.get("form", "") not in _FORMS:
                        continue
                    rows.append((concept, tag, prio, it.get("start"), it["end"], float(it["val"]), it["filed"],
                                 it.get("form"), it.get("accn", "")))

        for concept, tags in {**FLOW, **IDENTITY, **STOCK}.items():
            for prio, tag in enumerate(tags):
                take(concept, "us-gaap", tag, prio)
        for concept, (taxonomy, tag), _ in SHARE_SOURCES:
            take(concept, taxonomy, tag, 0)
    rows += _class_share_rows(path)
    df = pd.DataFrame(rows, columns=["concept", "tag", "prio", "start", "end", "val", "filed", "form", "accn"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["days"] = (df["end"] - df["start"]).dt.days
    return FactTable(df)


def _class_share_rows(path: Path) -> list[tuple]:
    """Share-count rows built from per-class XBRL facts (CLASS_SHARES), in load_facts' row layout."""
    sym = path.stem
    if sym not in CLASS_SHARES:
        return []
    f = path.parent.parent / "class_shares" / f"{sym}.json"
    if not f.exists():
        return []
    concept, xbrl, weights = CLASS_SHARES[sym]
    per: dict = {}
    for r in json.loads(f.read_text()):
        if r["concept"] == xbrl and r["member"] in weights and r["form"] in _FORMS:
            per.setdefault((r["accn"], r["start"], r["end"], r["filed"], r["form"]), {})[r["member"]] = r["val"]
    return [(concept, "class_shares", 0, start, end, sum(weights[m] * v[m] for m in weights), filed, form, accn)
            for (accn, start, end, filed, form), v in per.items() if set(v) == set(weights)]


_FLOW_SET, _STOCK_SET = set(FLOW) | set(IDENTITY), set(STOCK)
_SHARE_REF = {c: ref for c, _, ref in SHARE_SOURCES}
_EMPTY = pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _tag_quarters(latest: dict, by_filing: dict, vintages: dict) -> dict:
    """Fiscal-quarter values of one measure (a tag, or the merged facts of tags measuring the same thing; module
    doc): direct 3-month facts, then YTD differencing within a fiscal year, then FY minus three quarters.

    The subtrahend comes from the minuend's own filing when that filing reports it. Otherwise the latest vintages are
    differenced (an amendment of one YTD figure corrects it), except when the minuend was last re-reported as a
    comparative in a later year's filing (filed more than RECAST_DAYS after its period end): such comparatives may be
    re-based (AMT's FY2022 restated in 2024 for the India discontinued operation, while 9M 2022 never was), so the
    minuend vintage used is the first one filed on or after the subtrahend's."""
    q: dict = {}
    q_filed: dict = {}
    chains: dict = {}
    annual = []
    for (start, end), (val, accn, filed) in latest.items():
        days = (end - start).days
        if 80 <= days <= 100 and (end not in q or filed >= q_filed[end]):
            q[end], q_filed[end] = val, filed
        if 80 <= days <= 380:
            chains.setdefault(start, []).append((end, val, accn))
        if 350 <= days <= 380:
            annual.append((start, end, val))
    derived = {}
    for start, ch in chains.items():
        ch.sort()
        for k in range(1, len(ch)):
            e, v, a = ch[k]
            if not 80 <= (e - ch[k - 1][0]).days <= 100 or e in q:
                continue
            same = by_filing.get((start, ch[k - 1][0], a))
            if same is not None:
                derived[e] = v - same
                continue
            if (latest[(start, e)][2] - e).days > RECAST_DAYS:  # minuend last re-reported in a later year's filing
                sub_filed = latest[(start, ch[k - 1][0])][2]
                v = next((val for f, val in vintages[(start, e)] if f >= sub_filed), v)
            derived[e] = v - ch[k - 1][1]
    q.update(derived)
    for start, end, val in annual:
        if end in q:
            continue
        inside = [v for e, v in q.items() if start < e < end]
        if len(inside) == 3:
            q[end] = val - sum(inside)
    return q


def _fill_periods(primary: dict, other: dict) -> list:
    """Periods of `other` (not reported by the primary tag) that it may supply: `other` must measure the same thing
    as the primary around that period, i.e. >= SAME_MEASURE_SHARE of the periods both tags report and that end within
    NEAR_DAYS of it (else the NEAR_MIN nearest ones) agree within SERIES_AGREE_TOL; any period if the tags share none
    (a pure tag switch). T's total and continuing-operations cash flows agree except in the years of the
    WarnerMedia discontinued operation; PFE's total and contract revenue agree in 2020 but not in 2021-2023."""
    common = [k for k in other if k in primary]
    todo = [k for k in other if k not in primary]
    if not common or not todo:
        return todo
    agree = {k: abs(primary[k][0] - other[k][0]) <= SERIES_AGREE_TOL * max(abs(primary[k][0]), abs(other[k][0]), 1e-9)
             for k in common}
    out = []
    for k in todo:
        dist = sorted((abs((c[1] - k[1]).days), c) for c in common)
        near = [c for dd, c in dist if dd <= NEAR_DAYS] or [c for _, c in dist[:NEAR_MIN]]
        if np.mean([agree[c] for c in near]) >= SAME_MEASURE_SHARE:
            out.append(k)
    return out


class CompanyState:
    """Everything known about one company from filings ingested so far (filings are fed in date order, so a
    state built from filings with filed < d is exactly the information set at decision d).

    Per flow concept and tag it keeps the latest vintage of every period and every (period, filing) value, which is
    all that quarter derivation needs; stock items keep the latest instant; share sources keep their rows."""

    def __init__(self, ft: FactTable):
        d = ft.df.sort_values(["filed", "prio"], ascending=[True, False], kind="stable")
        self._rows = list(zip(d.concept, d.tag, d.start, d.end, d.val, d.filed, d.accn, d.prio, d.days))
        self._filed = pd.DatetimeIndex(d.filed)
        self._n = 0  # rows ingested
        self.latest: dict[str, dict] = {c: {} for c in _FLOW_SET}  # tag -> (start, end) -> (val, accn, filed)
        self.by_filing: dict[str, dict] = {c: {} for c in _FLOW_SET}  # tag -> (start, end, accn) -> val
        self.vintages: dict[str, dict] = {c: {} for c in _FLOW_SET}  # tag -> (start, end) -> [(filed, val)] by filed
        self.instant: dict[str, tuple] = {}  # concept -> (end, filed, -prio, val)
        self.shares: dict[str, list] = {c: [] for c in _SHARE_REF}  # rows (end, filed, days, val)
        self.last_filed: pd.Timestamp | None = None

    def ingest_before(self, asof: pd.Timestamp) -> "CompanyState":
        """Ingest every fact with filed < asof (idempotent; asof must not decrease)."""
        stop = int(self._filed.searchsorted(pd.Timestamp(asof), side="left"))
        for concept, tag, start, end, val, filed, accn, prio, days in self._rows[self._n:stop]:
            if concept in _FLOW_SET:
                if pd.isna(start):
                    continue
                self.latest[concept].setdefault(tag, {})[(start, end)] = (val, accn, filed)
                self.by_filing[concept].setdefault(tag, {})[(start, end, accn)] = val
                self.vintages[concept].setdefault(tag, {}).setdefault((start, end), []).append((filed, val))
            elif concept in _STOCK_SET:
                key = (end, filed, -prio, val)  # latest period, then the higher-priority tag, then the latest filing
                cur = self.instant.get(concept)
                if cur is None or (end, -prio, filed) >= (cur[0], cur[2], cur[1]):
                    self.instant[concept] = key
            elif val > 0:
                self.shares[concept].append((end, filed, 0 if pd.isna(days) else days, val))
            self.last_filed = filed
        self._n = max(self._n, stop)
        return self

    def tag_quarters(self, concept: str, tag: str) -> pd.Series:
        return pd.Series(self._tq(concept, tag), dtype=float).sort_index()

    def _tq(self, concept: str, tag: str) -> dict:
        return _tag_quarters(self.latest[concept].get(tag, {}), self.by_filing[concept].get(tag, {}),
                             self.vintages[concept].get(tag, {}))

    def primary_tag(self, concept: str) -> str | None:
        return self._merge(concept)[0]

    def quarters(self, concept: str) -> pd.Series:
        """Fiscal-quarter values of the concept, derived from the merged facts of the primary tag and the tags that
        measure the same thing (module doc)."""
        tag, merged = self._merge(concept)
        return pd.Series(_tag_quarters(*merged), dtype=float).sort_index() if tag else _EMPTY.copy()

    def _merge(self, concept: str) -> tuple[str | None, tuple[dict, dict, dict]]:
        """(primary tag, merged (latest, by_filing, vintages)). Primary: the first tag in priority order among those
        reporting the latest period end (for revenue, preferring tags that pass `revenue_is_total`). Every period
        takes the primary tag's facts if it reports that period, else those of the first tag (priority order) that
        measures the same thing as the primary around that period (`_fill_periods`); no other fact is used."""
        latest = self.latest[concept]
        tags = [t for t in FLOW.get(concept, IDENTITY.get(concept, [])) if latest.get(t)]
        if not tags:
            return None, ({}, {}, {})
        last_end = {t: max(e for _, e in latest[t]) for t in tags}
        current = [t for t in tags if last_end[t] == max(last_end.values())]
        if concept == "revenue" and len(current) > 1:
            current = [t for t in current if self.revenue_is_total(t) is True] or current
        primary = current[0]
        chosen = {k: primary for k in latest[primary]}  # period -> tag supplying it
        for t in tags:
            if t != primary:
                for k in _fill_periods(latest[primary], latest[t]):
                    chosen.setdefault(k, t)
        m_latest = {k: latest[t][k] for k, t in chosen.items()}
        m_vint = {k: self.vintages[concept][t][k] for k, t in chosen.items()}
        m_filing = {k: v for t in set(chosen.values()) for k, v in self.by_filing[concept][t].items()
                    if chosen.get(k[:2]) == t}
        return primary, (m_latest, m_filing, m_vint)

    def revenue_is_total(self, tag: str) -> bool | None:
        """Does `tag` satisfy revenue - costs = operating income in the latest annual filing that reports all three
        (costs = OperatingExpenses or CostsAndExpenses, with or without cost of revenue)? None if no filing does."""
        rev = self.by_filing["revenue"].get(tag, {})
        oi = self.by_filing["operating_income"].get("OperatingIncomeLoss", {})
        costs = [self.by_filing[c].get(t, {}) for c in IDENTITY for t in IDENTITY[c]]
        cogs = [self.by_filing["cost_of_revenue"].get(t, {}) for t in FLOW["cost_of_revenue"]]
        for k in sorted((k for k in rev if 350 <= (k[1] - k[0]).days <= 380 and k in oi), key=lambda k: k[1],
                        reverse=True):
            xs = [c[k] for c in costs if k in c]
            if not xs:
                continue
            gs = [0.0] + [c[k] for c in cogs if k in c]
            r, o = rev[k], oi[k]
            return any(abs(r - x - g - o) <= IDENTITY_TOL * abs(r) for x in xs for g in gs)
        return None

    def share_count(self, before: pd.Timestamp | None = None, source: str | None = None,
                    asof: pd.Timestamp | None = None) -> dict | None:
        """Latest share count: {shares, ref_date (split reference), end, source}. With `before`, only counts whose
        period ends on or before it (the count one year earlier); with `source`, that source only. No plausibility
        check (that needs split factors; see `pick_shares`)."""
        for concept, _, ref in SHARE_SOURCES:
            if source is not None and concept != source:
                continue
            rows = [r for r in self.shares[concept] if before is None or r[0] <= before]
            if not rows:
                continue
            end, filed, days, val = max(rows, key=lambda r: (r[0], r[1], -r[2]))  # latest period, shortest duration
            if before is None and asof is not None and (pd.Timestamp(asof) - filed).days > SHARES_MAX_AGE_DAYS:
                continue
            return {"shares": float(val), "ref_date": end if ref == "end" else filed, "end": end, "source": concept}
        return None

    def share_candidates(self) -> list[dict]:
        """Per share source (in preference order): its latest count with its filing date, the same source's previous
        report and its count about one year earlier (for `pick_shares` and the 1-year change). Staleness is left to
        the decision date."""
        key = lambda r: (r[0], r[1], -r[2])  # noqa: E731 - latest period, latest filing, shortest duration
        out = []
        for concept, _, ref in SHARE_SOURCES:
            rows = self.shares[concept]
            if not rows:
                continue
            refd = (lambda r: r[0]) if ref == "end" else (lambda r: r[1])
            cur = max(rows, key=key)
            prev = max((r for r in rows if r[0] < cur[0]), key=key, default=None)
            y1 = max((r for r in rows if r[0] <= cur[0] - pd.Timedelta(days=330)), key=key, default=None)
            if y1 is not None and (cur[0] - y1[0]).days > 400:
                y1 = None
            out.append({"source": concept, "shares": float(cur[3]), "ref_date": refd(cur), "filed": cur[1],
                        "prev_shares": None if prev is None else float(prev[3]),
                        "prev_ref_date": None if prev is None else refd(prev),
                        "y1_shares": None if y1 is None else float(y1[3]),
                        "y1_ref_date": None if y1 is None else refd(y1)})
        return out

    def snapshot(self, asof: pd.Timestamp) -> dict:
        """Point-in-time snapshot of the ingested filings (as-reported numbers; no prices). Nothing in it depends on
        `asof` except the `asof` field, so a snapshot taken right after a filing serves every later date until the
        next filing (FundamentalsPanel)."""
        out: dict = {"asof": pd.Timestamp(asof).date()}
        for c in FLOW:
            q = self.quarters(c)
            # require the last 4 quarters to be consecutive (<= ~13 months span) for TTM
            if len(q) >= 4 and (q.index[-1] - q.index[-4]).days <= 290:
                out[f"{c}_ttm"] = float(q.iloc[-4:].sum())
                out[f"{c}_last_q"] = float(q.iloc[-1])
                out[f"{c}_last_q_end"] = q.index[-1].date()
                if len(q) >= 8 and (q.index[-1] - q.index[-8]).days <= 670:
                    prev = float(q.iloc[-8:-4].sum())
                    out[f"{c}_ttm_yoy"] = (out[f"{c}_ttm"] / prev - 1.0) if prev > 0 else np.nan
                if len(q) >= 5 and (q.index[-1] - q.index[-5]).days <= 380:
                    prev_q = float(q.iloc[-5])
                    out[f"{c}_q_yoy"] = (out[f"{c}_last_q"] / prev_q - 1.0) if prev_q > 0 else np.nan
        if ("gross_profit_ttm" not in out and "revenue_ttm" in out and "cost_of_revenue_ttm" in out
                and out["revenue_last_q_end"] == out["cost_of_revenue_last_q_end"]):
            out["gross_profit_ttm"] = out["revenue_ttm"] - out["cost_of_revenue_ttm"]
        for c in STOCK:
            if c in self.instant:
                end, _, _, val = self.instant[c]
                out[c], out[f"{c}_end"] = float(val), end.date()
        out["share_counts"] = self.share_candidates()
        out["last_filing_date"] = None if self.last_filed is None else self.last_filed.date()
        return out


def quarterly_series(ft: FactTable, concept: str, asof: date) -> pd.Series:
    """Fiscal-quarter values of a flow concept known before `asof`, indexed by quarter end (see module doc)."""
    return CompanyState(ft).ingest_before(pd.Timestamp(asof)).quarters(concept)


def latest_instant(ft: FactTable, concept: str, asof: date) -> tuple[float, pd.Timestamp] | None:
    st = CompanyState(ft).ingest_before(pd.Timestamp(asof))
    return (float(st.instant[concept][3]), st.instant[concept][0]) if concept in st.instant else None


def share_count(ft: FactTable, asof: date, before: pd.Timestamp | None = None,
                source: str | None = None) -> dict | None:
    return CompanyState(ft).ingest_before(pd.Timestamp(asof)).share_count(before=before, source=source, asof=asof)


def snapshot(ft: FactTable, asof: date) -> dict:
    """Point-in-time fundamental snapshot using every fact with filed < asof (as-reported numbers; no prices).
    Semantic bucketing happens in the agent; price-dependent quantities come from `valuation`."""
    return CompanyState(ft).ingest_before(pd.Timestamp(asof)).snapshot(pd.Timestamp(asof))


def _ratio(a, b) -> float:
    return float(a) / float(b) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and b > 0 \
        else np.nan


def _in_band(x: float) -> bool:
    return SHARES_BAND[0] <= x <= SHARES_BAND[1]


def pick_shares(cands: list[dict], d: pd.Timestamp, split_factor) -> tuple[dict | None, list[str]]:
    """The share count used at d: the first source (preference order) whose count, filed at most
    SHARES_MAX_AGE_DAYS before d, lies within SHARES_BAND of at least one reference: another source's count not
    already rejected, or its own previous report, all put on d's split basis. A count with no reference at all is
    accepted (nothing to check it against). Returns (candidate or None, rejected sources); a rejection is a
    mis-scaled or garbled XBRL count."""
    d = pd.Timestamp(d)
    usable = [c for c in cands if (d - c["filed"]).days <= SHARES_MAX_AGE_DAYS]
    now = {c["source"]: c["shares"] * split_factor(c["ref_date"], d) for c in usable}
    rejected: list[str] = []
    for c in usable:
        refs = [v for s, v in now.items() if s != c["source"] and s not in rejected]  # a rejected count vetoes nothing
        if c["prev_shares"]:
            refs.append(c["prev_shares"] * split_factor(c["prev_ref_date"], d))
        x = now[c["source"]]
        if not refs or any(_in_band(x / r) for r in refs if r > 0):
            return c, rejected
        rejected.append(c["source"])
    return None, rejected


def valuation(snap: dict, raw_close: float | None, split_factor, d: pd.Timestamp, log: list | None = None) -> dict:
    """Price-dependent and ratio features at decision date d from a snapshot taken with filed < d.
    `raw_close`: last as-traded close on or before d; `split_factor(after, upto)`: cumulative split ratio. Rejected
    share counts and implausible 1-year share changes are appended to `log` (if given)."""
    g = snap.get
    d = pd.Timestamp(d)
    out = {
        "gross_margin": _ratio(g("gross_profit_ttm"), g("revenue_ttm")),
        "operating_margin": _ratio(g("operating_income_ttm"), g("revenue_ttm")),
        "net_margin": _ratio(g("net_income_ttm"), g("revenue_ttm")),
        "gross_profitability": _ratio(g("gross_profit_ttm"), g("assets")),
        "rev_growth_ttm": g("revenue_ttm_yoy", np.nan),
        "rev_growth_q_yoy": g("revenue_q_yoy", np.nan),
        "debt_to_assets": _ratio(g("lt_debt"), g("assets")),
        "cash_to_assets": _ratio(g("cash"), g("assets")),
        "market_value": np.nan, "earnings_yield": np.nan, "sales_yield": np.nan, "share_change_1y": np.nan,
    }
    c, rejected = pick_shares(g("share_counts") or [], d, split_factor)
    if rejected and log is not None:
        log.append(("rejected_share_count", d, rejected, c and c["source"]))
    if c is not None and c["y1_shares"]:
        chg = c["shares"] / (c["y1_shares"] * split_factor(c["y1_ref_date"], c["ref_date"]))
        if _in_band(chg):
            out["share_change_1y"] = chg - 1.0
        elif log is not None:
            log.append(("implausible_share_change_1y", d, [c["source"]], chg))
    if c is not None and raw_close is not None and np.isfinite(raw_close) and raw_close > 0:
        mv = raw_close * c["shares"] * split_factor(c["ref_date"], d)
        out["market_value"] = mv
        out["earnings_yield"] = g("net_income_ttm", np.nan) / mv
        out["sales_yield"] = g("revenue_ttm", np.nan) / mv
    if CASHFLOW_READY:
        fcf = g("cfo_ttm", np.nan) - g("capex_ttm", np.nan)
        out["fcf_yield"] = fcf / out["market_value"] if np.isfinite(out["market_value"]) else np.nan
        out["accruals"] = _ratio(g("net_income_ttm", np.nan) - g("cfo_ttm", np.nan), g("assets"))
        out["operating_cf_margin"] = _ratio(g("cfo_ttm"), g("revenue_ttm"))
        out["fcf_margin"] = _ratio(fcf, g("revenue_ttm"))
    return out


CODE_FEATURES = ("earnings_yield", "sales_yield", "gross_profitability", "operating_margin", "rev_growth_ttm",
                 "share_change_1y") + (("fcf_yield", "accruals") if CASHFLOW_READY else ())


class FundamentalsPanel:
    """Fast point-in-time fundamentals for many (symbol, date) pairs. A snapshot only changes when a filing arrives,
    so one snapshot is computed per distinct filing date f (from filings with filed <= f, by feeding a CompanyState
    in date order) and serves every decision date d in (f, next filing]. Everything date-dependent (prices, share
    staleness and plausibility) is computed at d by `valuation`, so the result equals the from-scratch
    `valuation(snapshot(ft, d), ...)` (tested)."""

    def __init__(self, facts_dir: Path, symbols: list[str], panels, start: str = "2016-01-01"):
        self.panels = panels  # data.prices.PricePanels
        self.snaps: dict[str, tuple[pd.DatetimeIndex, list[dict]]] = {}
        self.share_log: list[tuple] = []  # (symbol, kind, d, sources, detail), deduplicated per (symbol, kind, filing)
        self._logged: set = set()
        for s in symbols:
            p = Path(facts_dir) / f"{s}.json"
            if not p.exists():
                continue
            ft = load_facts(p)
            st = CompanyState(ft)
            filed = sorted(set(ft.df.filed[ft.df.filed >= pd.Timestamp(start)]))
            snaps = []
            for f in filed:
                asof = f + pd.Timedelta(days=1)
                snaps.append(st.ingest_before(asof).snapshot(asof))
            self.snaps[s] = (pd.DatetimeIndex(filed), snaps)

    def snapshot_asof(self, symbol: str, d: pd.Timestamp) -> dict | None:
        """The snapshot built from filings with filed < d."""
        got = self.snaps.get(symbol)
        if got is None:
            return None
        idx, snaps = got
        k = idx.searchsorted(pd.Timestamp(d).normalize(), side="left") - 1  # last filing date strictly before d
        return snaps[k] if k >= 0 else None

    def features(self, d: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
        d = pd.Timestamp(d)
        rows = {}
        for s in symbols:
            snap = self.snapshot_asof(s, d)
            if snap is None:
                continue
            rc = self.panels.raw_close_asof(s, d)
            events: list = []
            rows[s] = valuation(snap, rc[0] if rc else None,
                                lambda a, b, s=s: self.panels.splits.factor(s, a, b), d, log=events)
            for kind, day, sources, detail in events:
                if (s, kind, snap["last_filing_date"]) not in self._logged:
                    self._logged.add((s, kind, snap["last_filing_date"]))
                    self.share_log.append((s, kind, day, sources, detail))
                    _logger.warning("%s %s at %s: %s (%s)", s, kind, day.date(), sources, detail)
        return pd.DataFrame.from_dict(rows, orient="index")


def size_ranks(market_value: pd.Series, members: list[str]) -> dict[str, int]:
    """1 = largest raw market value within U_t (focal_company.size, design_v1 section 5.0). Names without a market
    value are ranked after all others, in the order of `members` (the universe's dollar-volume rank)."""
    mv = market_value.reindex(members)
    known = mv.dropna().sort_values(ascending=False, kind="mergesort").index.tolist()
    order = known + [s for s in members if s not in set(known)]
    return {s: k + 1 for k, s in enumerate(order)}
