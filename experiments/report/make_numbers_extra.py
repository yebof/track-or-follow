#!/usr/bin/env python3
"""DEV-12: registered macros the validity-first text needs (outline_v2 §5.2; positioning_v3 D-8) ->
paper/generated/numbers_extra.tex.

    .venv/bin/python experiments/report/make_numbers_extra.py [--check] [--list]

`main.tex` inputs this file right after generated/numbers.tex. It never edits make_numbers.py and never changes a
macro that make_numbers.py fills: a name already defined in numbers.tex is written here only if numbers.tex holds a
placeholder for it (the "wire" macros: \\preregDigest, \\nAdaptiveItems, \\maxAdaptiveQueries, the four FEED-1
shares), and then with \\renewcommand. Every other name is new.

Sources and guards (outline_v2 §5.2).
- Registered results: results/analysis/<name>.json through `make_numbers.registered_result` (registered window, lock
  open), exactly as make_numbers.py.
- The pre-registration's own records: results/prereg/power_prereg.json (NG CPU seconds per item), configs/prereg_v1.yaml
  (the pre-registration's SHA-256), docs/design/prereg_deviations.md (the count of `## DEV-` headings).
- Stated separate guards: results/pre/d3_pre.json (the W_pre-slice D3 record; appendix only) must say window
  W_pre_slice; data/processed/feed_provenance.json (FEED-1, outcome-free) must carry the SHA-256 of the current
  news table; results/judgments/verb_llm-verbalized_wclean_items2000/MANIFEST.json (VERB's paid spend of record).
A missing source leaves no placeholder behind silently: the script prints every missing macro and exits 1.

**Jev slot.** \\IfJevResults{<with Jev>}{<without>} takes its first argument, and the Jev macros carry values, only
when the descriptive Jev-panel output results/analysis/jev_rows.json exists and passes `_validity.jev_rows` (label,
lock, validated generations); otherwise every Jev macro expands to nothing (not an error). \\hTwoSpearmanJev,
\\hTwoCIJev, \\hTwoPlaceboJev and \\nJevUnits are the FORWARD post-release statistic (items created after Jev's
release, AR window realized); \\hTwoAUCEarningsJev is the earnings AUC on the registered H2 frame (W_clean,
contamination unverified). The \\jev... macros add the emu reference, agreement, other concepts, cost, latency and
the probes (lower bounds). Every Jev value is descriptive (experiments/jev_panel/jev_rows_spec.yaml).

DEV-12 (text for docs/design/prereg_deviations.md, to be appended by the orchestrator): "a new presentation script,
experiments/report/make_numbers_extra.py, writes paper/generated/numbers_extra.tex: registered-output macros that the
validity-first rewrite needs and make_numbers.py does not define (outline_v2 §5.2). It reads registered outputs only,
through make_numbers.registered_result, plus the stated records above; it edits no hashed file and computes no new
statistic (every value is a field of an existing output, rounded half-up). Touches a confirmatory statistic: no."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _validity as V  # noqa: E402
from make_numbers import _walk, fmt_num, fmt_p, half_up, registered_result  # noqa: E402  (read-only)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "generated" / "numbers_extra.tex"
NUMBERS = ROOT / "paper" / "generated" / "numbers.tex"
A = "results/analysis"
VERB_MANIFEST = ROOT / "results/judgments/verb_llm-verbalized_wclean_items2000/MANIFEST.json"
POWER = ROOT / "results/prereg/power_prereg.json"
PREREG_YAML = ROOT / "configs/prereg_v1.yaml"
DEVIATIONS = ROOT / "docs/design/prereg_deviations.md"
D3_PRE = ROOT / "results/pre/d3_pre.json"
FEED = ROOT / "data/processed/feed_provenance.json"
NEWS_TABLE = ROOT / "data/processed/news_table.parquet"

Getter = Callable[[], str | None]


# ------------------------------------------------------------------------------------------------ getters
def _fin(v) -> bool:
    import math

    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def val(name: str, *path, dec: int = 3, scale: float = 1.0, integer: bool = False, sign: bool = False) -> Getter:
    def get():
        v = _walk(registered_result(name), path)
        if not _fin(v):
            return None
        if integer:
            return fmt_num(int(round(v * scale)))
        s = half_up(v * scale, dec)
        return ("+" + s) if sign and v > 0 else s
    return get


def ci(name: str, *path, dec: int = 3, scale: float = 1.0) -> Getter:
    def get():
        v = _walk(registered_result(name), path)
        if not (isinstance(v, list) and len(v) == 2 and all(_fin(x) for x in v)):
            return None
        return f"[{half_up(v[0] * scale, dec)}, {half_up(v[1] * scale, dec)}]"
    return get


def positives(impl: str, concept: str) -> Getter:
    """Positives of an 8-K concept on a reader's units: prevalence x n of its score table."""
    def get():
        r = _walk(registered_result("h2"), ("per_implementation", impl, "sec_8k", concept)) or {}
        if not (_fin(r.get("prevalence")) and _fin(r.get("n"))):
            return None
        return fmt_num(int(round(r["prevalence"] * r["n"])))
    return get


def d2_row(level: str, question: str, period: str = "W_clean") -> dict | None:
    rows = _walk(registered_result("d2"), ("by_period_level",)) or []
    return next((r for r in rows if r.get("period") == period and r.get("level") == level
                 and r.get("question") == question), None)


def d2(level: str, question: str, field: str, dec: int = 0, scale: float = 1.0) -> Getter:
    def get():
        r = d2_row(level, question)
        v = None if r is None else r.get(field)
        return half_up(v * scale, dec) if _fin(v) else None
    return get


def d2_named(level: str) -> Getter:
    """Probes whose top answer named a firm (not cannot_tell): n x (1 - cannot_tell share)."""
    def get():
        r = d2_row(level, "reid_company")
        if r is None or not _fin(r.get("n")) or not _fin(r.get("cannot_tell_share")):
            return None
        return fmt_num(int(round(r["n"] * (1 - r["cannot_tell_share"]))))
    return get


def d2_chance(question: str) -> Getter:
    def get():
        v = _walk(registered_result("d2"), ("definitions", "chance", question))
        return fmt_num(v) if _fin(v) else None
    return get


def family(fam: str, field: str = "asr", dec: int = 1) -> Getter:
    def get():
        rows = _walk(registered_result("h1"), ("per_family",)) or []
        r = next((x for x in rows if x.get("kind") == "promotion" and x.get("reader") == "typed"
                  and x.get("family") == fam), None)
        if r is None:
            return None
        v = r.get(field)
        if field == "ci":
            return f"[{half_up(v[0] * 100, dec)}, {half_up(v[1] * 100, dec)}]" if isinstance(v, list) else None
        if field == "n":
            return fmt_num(int(v)) if _fin(v) else None
        return half_up(v * 100, dec) if _fin(v) else None
    return get


def contrast(reader: str, field: str) -> Getter:
    def get():
        rows = _walk(registered_result("h1"), ("secondary", "contrasts")) or []
        r = next((x for x in rows if x.get("reader") == reader and x.get("status") == "ok"), None)
        if r is None:
            return None
        if field == "estimate":
            v = r.get("estimate")
            return (("+" if v > 0 else "") + half_up(v * 100, 1)) if _fin(v) else None
        if field == "ci":
            v = r.get("ci")
            return f"[{half_up(v[0] * 100, 1)}, {half_up(v[1] * 100, 1)}]" if isinstance(v, list) else None
        if field == "p_holm":
            v = r.get("p_holm")
            if not _fin(v):
                return None
            return "\\ensuremath{<0.001}" if v < 0.001 else half_up(v, 3)
        return None
    return get


def rta(field: str, scale: float = 100.0, dec: int = 0) -> Getter:
    def get():
        d = registered_result("rta_ng")
        b = _walk(d, ("budgets",)) or {}
        if not b:
            return None
        top = str(max(int(k) for k in b))
        v = _walk(b, (top, field))
        return half_up(v * scale, dec) if _fin(v) else None
    return get


def rta_max_queries() -> str | None:
    b = _walk(registered_result("rta_ng"), ("budgets",)) or {}
    return fmt_num(max(int(k) for k in b)) if b else None


def rta_items() -> str | None:
    v = _walk(registered_result("rta_ng"), ("n_items",))
    return fmt_num(int(v)) if _fin(v) else None


def prereg_digest(full: bool = False) -> Getter:
    def get():
        if not PREREG_YAML.is_file():
            return None
        sha = (_walk(yaml.safe_load(PREREG_YAML.read_text()), ("preregistration", "sha256")) or "")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            return None
        if full:
            return "\\texttt{" + "\\allowbreak{}".join(sha[i:i + 16] for i in range(0, 64, 16)) + "}"
        return f"\\texttt{{{sha[:8]}}}\\ldots\\allowbreak\\texttt{{{sha[-5:]}}}"
    return get


def n_deviations() -> str | None:
    if not DEVIATIONS.is_file():
        return None
    return fmt_num(len(re.findall(r"^## DEV-\d+", DEVIATIONS.read_text(), flags=re.MULTILINE)))


def verb_cost() -> str | None:
    if not VERB_MANIFEST.is_file():
        return None
    v = json.loads(VERB_MANIFEST.read_text()).get("paid_usd_rows")
    return half_up(v, 2) if _fin(v) else None


def ng_seconds() -> str | None:
    if not POWER.is_file():
        return None
    v = _walk(json.loads(POWER.read_text()), ("cost_projection", "NG_cpu", "s_per_full_item_measured"))
    return half_up(v, 1) if _fin(v) else None


def d3(field: str) -> Getter:
    """results/pre/d3_pre.json (the W_pre-slice D3 record; appendix only): pooled delta_pre."""
    def get():
        if not D3_PRE.is_file():
            return None
        d = json.loads(D3_PRE.read_text())
        if d.get("window") != "W_pre_slice":
            return None
        v = _walk(d, ("pooled", "delta_pre", field))
        if field == "mean":
            return (("+" if v > 0 else "") + half_up(v, 4)) if _fin(v) else None
        return f"[{half_up(v[0], 3)}, {half_up(v[1], 3)}]" if isinstance(v, list) else None
    return get


_FEED_OK: dict = {}


def feed(cls: str | tuple[str, ...], dec: int = 1) -> Getter:
    """FEED-1 (outcome-free): byline-class share of the items the news programs read (scope company_le3) on W_clean,
    in %; the JSON must carry the SHA-256 of the current news table."""
    def get():
        if not FEED.is_file() or not NEWS_TABLE.is_file():
            return None
        d = json.loads(FEED.read_text())
        if "ok" not in _FEED_OK:
            h = hashlib.sha256()
            with open(NEWS_TABLE, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            _FEED_OK["ok"] = h.hexdigest() == d.get("table_sha256")
        if not _FEED_OK["ok"]:
            return None
        s = _walk(d, ("scopes", "company_le3", "W_clean")) or {}
        if cls == "items":
            return fmt_num(int(s["n_items"])) if _fin(s.get("n_items")) else None
        if cls == "outside":
            v = s.get("outside_reachable_share")
            return half_up(v * 100, dec) if _fin(v) else None
        shares = s.get("byline_class_share") or {}
        keys = (cls,) if isinstance(cls, str) else cls
        if not all(k in shares for k in keys):
            return None
        return half_up(100 * sum(shares[k] for k in keys), dec)
    return get


# ------------------------------------------------------------------------------------------------ round-1 additions
# DEV-13 (docs/design/prereg_deviations.md): registered-output macros added after the round-1 paper reviews
# (docs/reviews/paper_v2_r1_*.md). Every value is a field of an existing registered output or record, or a
# presentation of registered counts (an exact binomial or Wilson interval, a p bound, a realized MDE by the registered
# formula); none is a new test and none changes a registered decision.
SIMS = ROOT / "results/analysis/simulations/null_rates.json"
MECH = ROOT / "results/pre/mechanical_checks.json"


def rec(path: Path, *keys, dec: int = 3, scale: float = 1.0, integer: bool = False, sign: bool = False,
        window: str | None = None, script: str | None = None) -> Getter:
    """A field of a stated record outside results/analysis (pre-registration power output, pre-commit checks,
    simulations), guarded by its window or script field."""
    def get():
        if not path.is_file():
            return None
        d = json.loads(path.read_text())
        if window is not None and d.get("window") != window:
            return None
        if script is not None and _walk(d, ("meta", "script")) != script:
            return None
        v = _walk(d, keys)
        if not _fin(v):
            return None
        if integer:
            return fmt_num(int(round(v * scale)))
        out = half_up(v * scale, dec)
        return ("+" + out) if sign and v > 0 else out
    return get


def rec_ci(path: Path, *keys, dec: int = 3, scale: float = 1.0) -> Getter:
    def get():
        if not path.is_file():
            return None
        v = _walk(json.loads(path.read_text()), keys)
        if not (isinstance(v, list) and len(v) == 2 and all(_fin(x) for x in v)):
            return None
        return f"[{half_up(v[0] * scale, dec)}, {half_up(v[1] * scale, dec)}]"
    return get


def fam_reader(reader: str, fam: str, dec: int = 1) -> Getter:
    """Promotion ASR (%) of a family for any registered reader (h1.json per_family)."""
    def get():
        rows = _walk(registered_result("h1"), ("per_family",)) or []
        r = next((x for x in rows if x.get("kind") == "promotion" and x.get("reader") == reader
                  and x.get("family") == fam), None)
        return half_up(r["asr"] * 100, dec) if r and _fin(r.get("asr")) else None
    return get


def demotion(field: str, stratum: str | None = None, family_: str = "A8-1", dec: int = 0) -> Getter:
    """Registered demotion strata of the typed reader (h1.json demotion / pooled), in %."""
    def get():
        d = registered_result("h1") or {}
        if stratum is None:
            r = next((x for x in d.get("pooled") or [] if x.get("kind") == "demotion" and x.get("reader") == "typed"),
                     None)
        else:
            r = next((x for x in d.get("demotion") or [] if x.get("reader") == "typed" and x.get("stratum") == stratum
                      and x.get("family") == family_), None)
        if r is None:
            return None
        if field == "n":
            return fmt_num(int(r["n"])) if _fin(r.get("n")) else None
        v = r.get(field)
        if field == "ci":
            return f"[{half_up(v[0] * 100, dec)}, {half_up(v[1] * 100, dec)}]" if isinstance(v, list) else None
        return half_up(v * 100, dec) if _fin(v) else None
    return get


def cp_upper_zero(fam: str = "A1", level: float = 0.95) -> Getter:
    """Exact (Clopper-Pearson) two-sided upper bound (%) for a zero-success cell of the typed reader: 1 - (a/2)^(1/n).
    Presentation of a registered count; clustering is ignored (stated in the caption)."""
    def get():
        rows = _walk(registered_result("h1"), ("per_family",)) or []
        r = next((x for x in rows if x.get("kind") == "promotion" and x.get("reader") == "typed"
                  and x.get("family") == fam), None)
        if r is None or not _fin(r.get("n")) or r.get("asr") != 0:
            return None
        return half_up(100 * (1 - ((1 - level) / 2) ** (1 / r["n"])), 1)
    return get


def cp_interval(k_getter, n_getter, level: float = 0.95, dec: int = 1) -> Getter:
    def get():
        from scipy.stats import beta

        k, n = k_getter(), n_getter()
        if k is None or n is None or n <= 0:
            return None
        a = 1 - level
        lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
        return f"[{half_up(lo * 100, dec)}, {half_up(hi * 100, dec)}]"
    return get


def _holistic_counts():
    rows = _walk(registered_result("h1"), ("pooled",)) or []
    r = next((x for x in rows if x.get("kind") == "promotion" and x.get("reader") == "H"), None)
    if r is None:
        return None, None
    n = int(r["n"])
    return int(round(r["asr"] * n)), n


def wilson_cond(level: str) -> Getter:
    """Wilson 95% interval (%) of D2 accuracy among probes that named a firm (registered counts)."""
    def get():
        r = d2_row(level, "reid_company")
        if r is None or not _fin(r.get("n")) or not _fin(r.get("cannot_tell_share")):
            return None
        n = int(round(r["n"] * (1 - r["cannot_tell_share"])))
        k = int(r.get("correct", 0))
        if n <= 0:
            return None
        from math import sqrt

        z = 1.959963984540054
        c = (k + z * z / 2) / (n + z * z)
        h = z * sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
        return f"[{half_up(max(0, c - h) * 100, 0)}, {half_up(min(1, c + h) * 100, 0)}]"
    return get


def d2_ci(level: str, question: str = "reid_company", dec: int = 1) -> Getter:
    def get():
        r = d2_row(level, question)
        v = None if r is None else r.get("ci")
        return f"[{half_up(v[0] * 100, dec)}, {half_up(v[1] * 100, dec)}]" if isinstance(v, list) else None
    return get


def p_bound() -> str | None:
    """H2's Holm-adjusted p printed as a bound when it is below 10^-6 (the NW t-approximation is not reliable that far
    into the tail; the exact value stays in results/analysis/confirmatory.json)."""
    fam = _walk(registered_result("confirmatory"), ("family",)) or []
    r = next((x for x in fam if x.get("hypothesis") == "H2"), None)
    if r is None or not _fin(r.get("p_holm")):
        return None
    return "\\ensuremath{<10^{-6}}" if r["p_holm"] < 1e-6 else half_up(r["p_holm"], 3)


def realized_mde(name: str) -> Getter:
    """The registered MDE formula (z_holm x SE; power_prereg.json conventions) at the realized SE of a confirmatory
    statistic (as DEV-9 did for H1)."""
    def get():
        if not POWER.is_file():
            return None
        z = _walk(json.loads(POWER.read_text()), ("conventions", "z_holm"))
        se = _walk(registered_result(name), ("confirmatory", "se"))
        return half_up(z * se, 3) if _fin(z) and _fin(se) else None
    return get


def power_pct() -> str | None:
    """The power behind every registered MDE: Phi(z_holm - z_(1 - alpha/3)), alpha the family level (80%)."""
    if not POWER.is_file():
        return None
    from scipy.stats import norm

    z = _walk(json.loads(POWER.read_text()), ("conventions", "z_holm"))
    if not _fin(z):
        return None
    return half_up(100 * norm.cdf(z - norm.ppf(1 - 0.05 / 3)), 0)


def ln_two() -> str:
    import math

    return half_up(math.log(2), 4)


def lopo(which: str) -> Getter:
    """Registered PD2 leave-one-program-out with refit (X1): the smallest / largest weekly IC difference (primary minus
    the run without the program); printed only if every program's 95% interval covers zero (the sentence says so)."""
    def get():
        d = _walk(registered_result("pd2"), ("leave_one_out",)) or {}
        rows = [(k, v["delta_ic"]) for k, v in d.items() if v.get("status") == "ok"]
        if not rows or not all(r["ci"][0] <= 0 <= r["ci"][1] for _, r in rows):
            return None
        k, r = (min if which in ("min", "minprog") else max)(rows, key=lambda x: x[1]["mean"])
        if which.endswith("prog"):
            return k
        v = r["mean"]
        return ("+" if v > 0 else "") + half_up(v, 3)
    return get


def h3_reversed() -> str | None:
    """DEV-13 (as DEV-9 for H1): a descriptive clause used when H3's registered one-sided test did not reject AND its
    95% CI lies entirely below 0 (the side opposite to the claim); empty otherwise. Estimate and CI only."""
    d = registered_result("confirmatory") or {}
    r = next((x for x in d.get("family") or [] if x.get("hypothesis") == "H3"), None)
    if r is None or r.get("claim") != "fallback" or not isinstance(r.get("ci"), list):
        return None
    if not r["ci"][1] < 0:
        return ""
    return ("The estimate is negative and its \\ciLevel\\% interval lies below zero: in this system the typed "
            "judgments lowered the pooled logit's rank IC. The registered test was one-sided for an increment, so "
            "this is an estimate, not a tested harm.")


def abstract_h3() -> str | None:
    d = registered_result("confirmatory") or {}
    r = next((x for x in d.get("family") or [] if x.get("hypothesis") == "H3"), None)
    if r is None:
        return None
    if r.get("claim") == "stated":
        return "the judgments add rank information over code-only features in the confirmatory test"
    return "the confirmatory test detects no rank information over code-only features"


def abstract_h1() -> str | None:
    d = registered_result("confirmatory") or {}
    r = next((x for x in d.get("family") or [] if x.get("hypothesis") == "H1"), None)
    if r is None:
        return None
    if r.get("claim") == "stated":
        return ("reading items one at a time with guard questions lowered attack success below a holistic reader's "
                "(\\hOneASRTyped\\% vs.\\ \\hOneASRHolistic\\%)")
    rev = isinstance(r.get("ci"), list) and r["ci"][0] > 0
    return ("reading items one at a time with guard questions did not lower attack success below a holistic "
            "reader's" + (" (\\hOneASRTyped\\% vs.\\ \\hOneASRHolistic\\%)" if rev else ""))


def prose_date(which: str) -> Getter:
    """A window date of configs/system_v1.yaml (the one make_numbers.py prints in ISO form) as prose, e.g. 5 July 2024
    (editor review L3)."""
    def get():
        import datetime as dt

        system = yaml.safe_load((ROOT / "configs/system_v1.yaml").read_text())
        w = system.get("windows") or {}
        v = {"clean_start": (w.get("W_clean") or [None])[0], "clean_end": (w.get("W_clean") or [None, None])[1],
             "release_start": (w.get("W_release") or [None])[0]}.get(which)
        if not v:
            return None
        d = dt.date.fromisoformat(str(v))
        return f"{d.day}~{d.strftime('%B')} {d.year}"
    return get


def month_prose(ym: str | None) -> str | None:
    import datetime as dt

    if not ym:
        return None
    d = dt.date.fromisoformat(f"{ym}-01")
    return f"{d.strftime('%B')} {d.year}"


def d1_month(which: str) -> Getter:
    def get():
        d = registered_result("d1") or {}
        months = _walk(d, ("schedule", "months")) or []
        if which == "first":
            return month_prose(months[0] if months else None)
        if which == "last":
            return month_prose(months[-1] if months else None)
        if which == "centre":
            return month_prose(_walk(d, ("settings", "center")))
        if which == "windows":
            w = d.get("monthly_windows") or []
            return fmt_num(len(w)) if w else None
        return None
    return get


def emu_cutoff_month() -> str | None:
    """The documented knowledge cutoff of emu's LLM from the archived vendor page (as table_models.py reads it)."""
    import datetime as dt

    page = ROOT / "docs/pricing/openai_model_gpt-4.1-mini.html"
    if not page.is_file():
        return None
    m = re.search(r"([A-Z][a-z]{2} \d{2}, \d{4}) knowledge cutoff", page.read_text(errors="replace"))
    if not m:
        return None
    d = dt.datetime.strptime(m.group(1), "%b %d, %Y").date()
    return f"{d.strftime('%B')} {d.year}"


def n_secondary() -> str | None:
    v = _walk(registered_result("h1"), ("secondary", "m_registered"))
    return fmt_num(int(v)) if _fin(v) else None


def n_families(kind: str) -> Getter:
    def get():
        rows = _walk(registered_result("h1"), ("per_family",)) or []
        fams = {r["family"] for r in rows if r.get("kind") == kind and r.get("reader") == "typed"}
        return fmt_num(len(fams)) if fams else None
    return get


def lit_search_date() -> str | None:
    """The date of the last literature search behind the gap sentence (LIT-1 log), as prose."""
    import datetime as dt

    f = ROOT / "literature/meta/lit1_search_log.json"
    if not f.is_file():
        return None
    v = json.loads(f.read_text()).get("date_utc")
    if not v:
        return None
    d = dt.date.fromisoformat(v[:10])
    return f"{d.day}~{d.strftime('%B')} {d.year}"


def h2_sentence_bound() -> str | None:
    """DEV-13: the registered H2 sentence with its Holm-adjusted p printed as a bound (\\hTwoPholmBound) instead of
    3.3 x 10^-20 (stats review minor 4). The wording is make_numbers.py's own registered template for the realized
    outcome; only the p format changes."""
    import make_numbers as MN

    d = registered_result("confirmatory") or {}
    claims = d.get("claims") or {}
    if "H2" not in claims:
        return None
    sent = next(x for x in MN.SENTENCES if x.name == "hTwoSentence")
    text = sent.confirmed if claims["H2"] else sent.fallback
    return text.replace("Holm-adjusted $p = \\hTwoPholm$", "Holm-adjusted $p$ \\hTwoPholmBound{}")


def pd1_row(book: str, *keys, dec: int = 1, scale: float = 100.0) -> Getter:
    def get():
        v = _walk(registered_result("pd1"), ("rows", book, *keys))
        if isinstance(v, list):
            return f"[{half_up(v[0] * scale, dec)}, {half_up(v[1] * scale, dec)}]" if all(_fin(x) for x in v) else None
        return half_up(v * scale, dec) if _fin(v) else None
    return get


# ------------------------------------------------------------------------------------------------ Jev slot
JR = "results/analysis/jev_rows.json"


def _jv(*path):
    d = V.jev_rows()
    return None if d is None else _walk(d, path)


def jev_val(*path, dec: int = 3, scale: float = 1.0, integer: bool = False) -> Getter:
    """A value of the descriptive Jev panel (results/analysis/jev_rows.json), or "" (the slot prints nothing) when
    no panel output exists. A panel output with the field missing is an error (None: reported as missing)."""
    def get():
        if V.jev_rows() is None:
            return ""
        v = _jv(*path)
        if not _fin(v):
            return None
        return fmt_num(int(round(v * scale))) if integer else half_up(v * scale, dec)
    return get


def jev_ci(*path, dec: int = 3, scale: float = 1.0) -> Getter:
    def get():
        if V.jev_rows() is None:
            return ""
        v = _jv(*path)
        if not (isinstance(v, list) and len(v) == 2 and all(_fin(x) for x in v)):
            return None
        return f"[{half_up(v[0] * scale, dec)}, {half_up(v[1] * scale, dec)}]"
    return get


def jev_d2(level: str, question: str, field: str = "accuracy", who: str = "jev") -> Getter:
    """D2 re-identification accuracy (percent) of a level and question, from the panel's probe rows."""
    def get():
        if V.jev_rows() is None:
            return ""
        key = "emu_reference_registered" if who == "emu" else "jev"
        rows = _jv("probes", "d2", key, "pooled_by_level") or []
        r = next((x for x in rows if x.get("level") == level and x.get("question") == question), None)
        v = (r or {}).get(field)
        return half_up(v * 100, 1) if _fin(v) else None
    return get


def jev_flag() -> str:
    return "#1" if V.jev_rows() is not None else "#2"


def jev_forward_post_release() -> bool | None:
    """True if every forward unit of the panel was created after Jev's release (the generation's labels); None when
    no panel output exists. Read from the validated forward generation's MANIFEST labels (meta.generations)."""
    d = V.jev_rows()
    if d is None:
        return None
    labels = _walk(d, ("meta", "generations", "forward", "labels")) or {}
    return set(labels) == {"post-release"}


def jev_text(kind: str) -> Getter:
    """Generated wording of the Jev slot (empty without the panel output). The panel's status is descriptive
    (pre-registration: Jev rows are descriptive; jev_rows_spec.yaml); the timing and recall phrases describe the
    FORWARD units that the slot's validity numbers come from."""
    def get():
        post = jev_forward_post_release()
        if post is None:
            return ""
        if not post:
            raise SystemExit(f"{JR}: a forward unit is not post-release; the Jev slot wording does not apply")
        return {"status": "descriptive",
                "timing": "on items published after Jev's release",
                "recall": "these items were published after Jev's release, so recall in its weights cannot reach "
                          "them"}[kind]
    return get


# ------------------------------------------------------------------------------------------------ the list
H2 = "h2"
P = "per_implementation"
E = "earnings_results_vs_2.02"
DL = "deal_or_investment_vs_1.01/2.01"
MG = "management_or_labor_vs_5.02"
MACROS: list[tuple[str, str, Getter, str]] = [
    # --- H2 confirmatory frame and controls
    ("nHTwoUnits", "H2 confirmatory units (items)", val(H2, "confirmatory", "n_units", integer=True),
     f"{A}/h2.json confirmatory.n_units"),
    ("nHTwoFirmDays", "distinct (focal firm, event session) pairs of the H2 frame",
     val(H2, "confirmatory", "distinct_focal_d0", integer=True), f"{A}/h2.json confirmatory.distinct_focal_d0"),
    ("hTwoT", "H2 Newey-West(4) t (text prints t, not p; positioning_v3 T-m5)", val(H2, "confirmatory", "t", dec=1),
     f"{A}/h2.json confirmatory.t"),
    ("hTwoCIRelease", "95% CI of H2 on W_release", ci(H2, "sensitivities", "W_release", "ci"),
     f"{A}/h2.json sensitivities.W_release.ci"),
    ("nHTwoUnitsRelease", "H2 units on W_release", val(H2, "sensitivities", "W_release", "n_units", integer=True),
     f"{A}/h2.json sensitivities.W_release.n_units"),
    ("hTwoPlaceboEmuCI", "95% CI of the placebo AR[-5,-1] weekly mean (emu)", ci(H2, "placebo_ar_m5_m1", "ci"),
     f"{A}/h2.json placebo_ar_m5_m1.ci"),
    ("hTwoCIPreOpen", "95% CI of the pre-open control", ci(H2, "controls", "pre_open_only", "ci"),
     f"{A}/h2.json controls.pre_open_only.ci"),
    ("nHTwoUnitsPreOpen", "units of the pre-open control", val(H2, "controls", "pre_open_only", "n_units",
                                                                 integer=True),
     f"{A}/h2.json controls.pre_open_only.n_units"),
    ("hTwoPriorLookExcl", "H2 with the declared prior-look stock-weeks excluded",
     val(H2, "controls", "prior_look_excluded", "mean"), f"{A}/h2.json controls.prior_look_excluded.mean"),
    ("hTwoPriorLookExclCI", "its 95% CI", ci(H2, "controls", "prior_look_excluded", "ci"),
     f"{A}/h2.json controls.prior_look_excluded.ci"),
    ("hTwoDropThirteen", "H2 with the first 13 weeks dropped", val(H2, "sensitivities", "drop_first_13", "mean"),
     f"{A}/h2.json sensitivities.drop_first_13.mean"),
    ("hTwoDropThirteenCI", "its 95% CI", ci(H2, "sensitivities", "drop_first_13", "ci"),
     f"{A}/h2.json sensitivities.drop_first_13.ci"),
    ("hTwoCIEmuPooled", "week-block 95% CI of the pooled emu Spearman",
     ci(H2, P, "emu", "spearman_ar_pooled", "ci_week_block"), f"{A}/h2.json {P}.emu.spearman_ar_pooled.ci_week_block"),
    # --- descriptive readers
    ("hTwoCIVerb", "week-block 95% CI of VERB's pooled Spearman", ci(H2, P, "VERB", "spearman_ar", "ci_week_block"),
     f"{A}/h2.json {P}.VERB.spearman_ar.ci_week_block"),
    ("hTwoCING", "week-block 95% CI of NG's pooled Spearman", ci(H2, P, "NG", "spearman_ar", "ci_week_block"),
     f"{A}/h2.json {P}.NG.spearman_ar.ci_week_block"),
    ("hTwoPlaceboVerb", "VERB pooled placebo Spearman", val(H2, P, "VERB", "spearman_ar_placebo", "rho"),
     f"{A}/h2.json {P}.VERB.spearman_ar_placebo.rho"),
    ("hTwoPlaceboVerbCI", "its week-block 95% CI", ci(H2, P, "VERB", "spearman_ar_placebo", "ci_week_block"),
     f"{A}/h2.json {P}.VERB.spearman_ar_placebo.ci_week_block"),
    ("hTwoPlaceboNG", "NG pooled placebo Spearman", val(H2, P, "NG", "spearman_ar_placebo", "rho"),
     f"{A}/h2.json {P}.NG.spearman_ar_placebo.rho"),
    ("hTwoPlaceboNGCI", "its week-block 95% CI", ci(H2, P, "NG", "spearman_ar_placebo", "ci_week_block"),
     f"{A}/h2.json {P}.NG.spearman_ar_placebo.ci_week_block"),
    ("hTwoEmuOnVerb", "emu pooled Spearman on VERB's units", val(H2, P, "VERB", "emu_same_units", "rho"),
     f"{A}/h2.json {P}.VERB.emu_same_units.rho"),
    ("hTwoEmuOnVerbCI", "its week-block 95% CI", ci(H2, P, "VERB", "emu_same_units", "ci_week_block"),
     f"{A}/h2.json {P}.VERB.emu_same_units.ci_week_block"),
    ("hTwoEmuOnNG", "emu pooled Spearman on NG's units", val(H2, P, "NG", "emu_same_units", "rho"),
     f"{A}/h2.json {P}.NG.emu_same_units.rho"),
    ("hTwoEmuOnNGCI", "its week-block 95% CI", ci(H2, P, "NG", "emu_same_units", "ci_week_block"),
     f"{A}/h2.json {P}.NG.emu_same_units.ci_week_block"),
    ("agreeVerbRho", "Spearman of E[effect], emu vs VERB", val(H2, P, "VERB", "agreement", "spearman_E_effect", dec=2),
     f"{A}/h2.json {P}.VERB.agreement.spearman_E_effect"),
    ("agreeVerbKappa", "Cohen's kappa of top event types, emu vs VERB",
     val(H2, P, "VERB", "agreement", "top_event_type", "kappa", dec=2),
     f"{A}/h2.json {P}.VERB.agreement.top_event_type.kappa"),
    ("agreeNGRho", "Spearman of E[effect], emu vs NG", val(H2, P, "NG", "agreement", "spearman_E_effect", dec=2),
     f"{A}/h2.json {P}.NG.agreement.spearman_E_effect"),
    ("agreeNGKappa", "Cohen's kappa of top event types, emu vs NG",
     val(H2, P, "NG", "agreement", "top_event_type", "kappa", dec=2),
     f"{A}/h2.json {P}.NG.agreement.top_event_type.kappa"),
    ("nVerbUnits", "VERB sample units", val(H2, "implementation_frames", "VERB", "units", integer=True),
     f"{A}/h2.json implementation_frames.VERB.units"),
    ("nNGUnits", "NG sample units", val(H2, "implementation_frames", "NG", "units", integer=True),
     f"{A}/h2.json implementation_frames.NG.units"),
    ("verbEligible", "VERB's own C2-eligible units", val(H2, "implementation_frames", "VERB", "c2_eligible",
                                                           integer=True),
     f"{A}/h2.json implementation_frames.VERB.c2_eligible"),
    ("ngEligible", "NG's own C2-eligible units", val(H2, "implementation_frames", "NG", "c2_eligible", integer=True),
     f"{A}/h2.json implementation_frames.NG.c2_eligible"),
    # --- 8-K concept agreement (registered secondary)
    ("hTwoAUCEarningsEmu", "AUC of P(earnings_results) vs an Item 2.02 8-K, emu", val(H2, "sec_8k", E, "auc"),
     f"{A}/h2.json sec_8k.{E}.auc"),
    ("hTwoAUCEarningsEmuCI", "its week-block 95% CI", ci(H2, "sec_8k", E, "auc_ci"), f"{A}/h2.json sec_8k.{E}.auc_ci"),
    ("hTwoAUCEarningsVerb", "the same, VERB", val(H2, P, "VERB", "sec_8k", E, "auc"),
     f"{A}/h2.json {P}.VERB.sec_8k.{E}.auc"),
    ("hTwoAUCEarningsVerbCI", "its week-block 95% CI", ci(H2, P, "VERB", "sec_8k", E, "auc_ci"),
     f"{A}/h2.json {P}.VERB.sec_8k.{E}.auc_ci"),
    ("hTwoAUCEarningsNG", "the same, NG", val(H2, P, "NG", "sec_8k", E, "auc"), f"{A}/h2.json {P}.NG.sec_8k.{E}.auc"),
    ("hTwoAUCEarningsNGCI", "its week-block 95% CI", ci(H2, P, "NG", "sec_8k", E, "auc_ci"),
     f"{A}/h2.json {P}.NG.sec_8k.{E}.auc_ci"),
    ("hTwoAUCDealEmu", "AUC of P(deal_or_investment) vs Items 1.01/2.01, emu", val(H2, "sec_8k", DL, "auc"),
     f"{A}/h2.json sec_8k.{DL}.auc"),
    ("hTwoAUCDealEmuCI", "its week-block 95% CI", ci(H2, "sec_8k", DL, "auc_ci"), f"{A}/h2.json sec_8k.{DL}.auc_ci"),
    ("hTwoAUCMgmtEmu", "AUC of P(management_or_labor) vs Item 5.02, emu", val(H2, "sec_8k", MG, "auc"),
     f"{A}/h2.json sec_8k.{MG}.auc"),
    ("hTwoAUCMgmtEmuCI", "its week-block 95% CI", ci(H2, "sec_8k", MG, "auc_ci"), f"{A}/h2.json sec_8k.{MG}.auc_ci"),
    ("hTwoAUCDealNG", "AUC of the deal concept, NG (its CI covers 0.5: the SEC clause is not stated)",
     val(H2, P, "NG", "sec_8k", DL, "auc"), f"{A}/h2.json {P}.NG.sec_8k.{DL}.auc"),
    ("hTwoAUCDealNGPositives", "Items 1.01/2.01 positives among NG's units", positives("NG", DL),
     f"{A}/h2.json {P}.NG.sec_8k.{DL}.prevalence x n"),
    ("hTwoECEEarningsEmu", "ECE of P(earnings_results) vs Item 2.02, emu", val(H2, "sec_8k", E, "ece"),
     f"{A}/h2.json sec_8k.{E}.ece"),
    ("hTwoVsExpAUCEmu", "AUC of P(above)/(P(above)+P(below)) vs 1[AR[0,+1] > 0], emu",
     val(H2, "vs_expectations_vs_ar_sign", "auc"), f"{A}/h2.json vs_expectations_vs_ar_sign.auc"),
    ("hTwoVsExpNEmu", "units of that AUC (top vs_expectations above or below), emu",
     val(H2, "vs_expectations_vs_ar_sign", "n", integer=True), f"{A}/h2.json vs_expectations_vs_ar_sign.n"),
    ("sentinelAUC", "sentinel h vs next-week tail events, AUC (App. C only)",
     val(H2, "agent_level", "sentinel_h_vs_T3", "auc"), f"{A}/h2.json agent_level.sentinel_h_vs_T3.auc"),
    ("sentinelAUCCI", "its week-block 95% CI", ci(H2, "agent_level", "sentinel_h_vs_T3", "ci_week_block"),
     f"{A}/h2.json agent_level.sentinel_h_vs_T3.ci_week_block"),
    # --- D2 re-identification (W_clean) and the D1 rule
    ("dTwoCompanyFull", "top answer = the true firm, full blinding (%)", d2("full", "reid_company", "accuracy",
                                                                           scale=100),
     f"{A}/d2.json by_period_level[W_clean, full, reid_company].accuracy"),
    ("dTwoCompanyNames", "the same, names masked only (%)", d2("names", "reid_company", "accuracy", scale=100),
     f"{A}/d2.json by_period_level[W_clean, names, reid_company].accuracy"),
    ("dTwoCompanyNone", "the same, unblinded (%)", d2("none", "reid_company", "accuracy", scale=100),
     f"{A}/d2.json by_period_level[W_clean, none, reid_company].accuracy"),
    ("dTwoCannotTell", "cannot_tell share, full blinding (%)", d2("full", "reid_company", "cannot_tell_share",
                                                                  scale=100),
     f"{A}/d2.json by_period_level[W_clean, full, reid_company].cannot_tell_share"),
    ("dTwoMeanPTrueFull", "mean probability on the true firm, full blinding",
     d2("full", "reid_company", "mean_p_true", dec=3), f"{A}/d2.json by_period_level[W_clean, full].mean_p_true"),
    ("dTwoNFull", "company probes per blinding level (W_clean)", d2("full", "reid_company", "n"),
     f"{A}/d2.json by_period_level[W_clean, full, reid_company].n"),
    ("dTwoNamedFull", "probes that named a firm, full blinding", d2_named("full"),
     f"{A}/d2.json n x (1 - cannot_tell_share) [W_clean, full]"),
    ("dTwoNamedCorrectFull", "of which the true firm, full blinding", d2("full", "reid_company", "correct"),
     f"{A}/d2.json by_period_level[W_clean, full, reid_company].correct"),
    ("dTwoNamedNames", "probes that named a firm, names masked", d2_named("names"),
     f"{A}/d2.json n x (1 - cannot_tell_share) [W_clean, names]"),
    ("dTwoNamedCorrectNames", "of which the true firm, names masked", d2("names", "reid_company", "correct"),
     f"{A}/d2.json by_period_level[W_clean, names, reid_company].correct"),
    ("dTwoYearFull", "top answer = the true year, full blinding (%)", d2("full", "reid_year", "accuracy", dec=1,
                                                                        scale=100),
     f"{A}/d2.json by_period_level[W_clean, full, reid_year].accuracy"),
    ("dTwoYearN", "year probes, full blinding", d2("full", "reid_year", "n"),
     f"{A}/d2.json by_period_level[W_clean, full, reid_year].n"),
    ("dTwoChanceCompany", "chance of the company probe (1 of 5 options)", d2_chance("reid_company"),
     f"{A}/d2.json definitions.chance.reid_company"),
    ("dTwoChanceYear", "chance of the year probe", d2_chance("reid_year"), f"{A}/d2.json definitions.chance.reid_year"),
    ("dOneCentreAUC", "D1 AUC of the registered centre window", val("d1", "rule", "center_window", "auc"),
     f"{A}/d1.json rule.center_window.auc"),
    ("dOneCentreP", "its name-permutation p (the registered rule)", val("d1", "rule", "center_window", "p_perm"),
     f"{A}/d1.json rule.center_window.p_perm"),
    ("dOneCentreN", "probes in the centre window", val("d1", "rule", "center_window", "n", integer=True),
     f"{A}/d1.json rule.center_window.n"),
    ("dThreePre", "D3 pooled Delta_pre (blinded - unblinded IC), W_pre slice (App. D)", d3("mean"),
     "results/pre/d3_pre.json pooled.delta_pre.mean"),
    ("dThreePreCI", "its block-bootstrap 95% CI", d3("ci95_block_bootstrap"),
     "results/pre/d3_pre.json pooled.delta_pre.ci95_block_bootstrap"),
    # --- planted content (H1 family) and the bound
    ("localizationCI", "95% CI of the typed localization rate (%)",
     ci("localization", "by_reader", "typed", "ci", dec=1, scale=100),
     f"{A}/localization.json by_reader.typed.ci"),
    ("nodefDelta", "ASR typed - typed without defenses (pp)", contrast("nodef", "estimate"),
     f"{A}/h1.json secondary.contrasts[nodef].estimate"),
    ("nodefCI", "its 95% CI (pp)", contrast("nodef", "ci"), f"{A}/h1.json secondary.contrasts[nodef].ci"),
    ("nodefPholm", "its Holm-adjusted p over the seven registered secondary contrasts", contrast("nodef", "p_holm"),
     f"{A}/h1.json secondary.contrasts[nodef].p_holm"),
    ("hguardDelta", "ASR typed - holistic with guards (pp)", contrast("H-guard", "estimate"),
     f"{A}/h1.json secondary.contrasts[H-guard].estimate"),
    ("hguardCI", "its 95% CI (pp)", contrast("H-guard", "ci"), f"{A}/h1.json secondary.contrasts[H-guard].ci"),
    *[(f"asrFam{w}", f"typed ASR (%), family {f}", family(f), f"{A}/h1.json per_family[typed, {f}].asr")
      for f, w in (("A1", "Aone"), ("A2", "Atwo"), ("A3", "Athree"), ("A4", "Afour"), ("A5", "Afive"),
                   ("A6", "Asix"), ("A7", "Aseven"))],
    *[(f"asrFam{w}CI", f"its week-clustered 95% CI (%), family {f}", family(f, "ci"),
       f"{A}/h1.json per_family[typed, {f}].ci") for f, w in (("A2", "Atwo"), ("A6", "Asix"), ("A7", "Aseven"))],
    *[(f"asrFam{w}N", f"attacked units, family {f}", family(f, "n"), f"{A}/h1.json per_family[typed, {f}].n")
      for f, w in (("A1", "Aone"), ("A5", "Afive"), ("A7", "Aseven"))],
    ("rtaNGSurvival", "RT-A-NG: items with a surviving edit at the largest budget (%)", rta("survival_rate"),
     f"{A}/rta_ng.json budgets[max].survival_rate"),
    ("rtaNGUp", "RT-A-NG: items moved one effect level up at the largest budget (%)",
     rta("share_survivor_effect_up_1_level"), f"{A}/rta_ng.json budgets[max].share_survivor_effect_up_1_level"),
    ("nAdaptiveItems", "RT-A-NG items (wire)", rta_items, f"{A}/rta_ng.json n_items"),
    ("maxAdaptiveQueries", "RT-A-NG largest query budget (wire)", rta_max_queries, f"{A}/rta_ng.json budgets (max key)"),
    ("hThreeCIRelease", "95% CI of H3 on W_release", ci("h3", "sensitivities", "W_release", "ci"),
     f"{A}/h3.json sensitivities.W_release.ci"),
    # --- pre-specification and cost
    ("preregDigest", "SHA-256 of the pre-registration, short form (wire)", prereg_digest(),
     "configs/prereg_v1.yaml preregistration.sha256"),
    ("preregDigestFull", "SHA-256 of the pre-registration, full", prereg_digest(full=True),
     "configs/prereg_v1.yaml preregistration.sha256"),
    ("nDeviations", "logged deviations (count of '## DEV-' headings)", n_deviations,
     "docs/design/prereg_deviations.md"),
    ("costPerItem", "$ per stock-week per news item (x 10^-3)", val("pd0", "scaling", "usd_per_stock_week_per_item",
                                                                   scale=1e3),
     f"{A}/pd0.json scaling.usd_per_stock_week_per_item"),
    ("verbCost", "VERB's paid spend of record ($)", verb_cost, "VERB MANIFEST.json paid_usd_rows"),
    ("ngSecondsPerItem", "NG CPU seconds per full item (one 6-thread worker)", ng_seconds,
     "results/prereg/power_prereg.json cost_projection.NG_cpu.s_per_full_item_measured"),
    # --- FEED-1 (outcome-free; wire)
    ("feedDeskShare", "desk bylines, items the news programs read on W_clean (%)", feed("desk"),
     "data/processed/feed_provenance.json scopes.company_le3.W_clean"),
    ("feedContribShare", "named contributors (%)", feed("named"),
     "data/processed/feed_provenance.json scopes.company_le3.W_clean"),
    ("feedWireShare", "wire or release bylines (%)", feed("wire"),
     "data/processed/feed_provenance.json scopes.company_le3.W_clean"),
    ("feedOtherShare", "other bylines (%)", feed(("other_email", "other_org", "other_empty")),
     "data/processed/feed_provenance.json scopes.company_le3.W_clean"),
    ("feedItems", "items the news programs could read on W_clean", feed("items"),
     "data/processed/feed_provenance.json scopes.company_le3.W_clean.n_items"),
    ("feedOutsideShare", "items whose byline an outside author can obtain (%)", feed("outside"),
     "data/processed/feed_provenance.json scopes.company_le3.W_clean.outside_reachable_share"),
    # --- round-1 additions (DEV-13)
    ("nItemRowsWindow", "item-firm rows of the news programs in the H2 window (before eligibility)",
     val(H2, "frame", "items_in_window", integer=True), f"{A}/h2.json frame.items_in_window"),
    ("nKeptRows", "item-firm rows kept (about the focal firm, a new development, not quarantined)",
     val(H2, "frame", "kept", integer=True), f"{A}/h2.json frame.kept"),
    ("hTwoPholmBound", "H2 Holm-adjusted p as a bound (DEV-13)", p_bound, f"{A}/confirmatory.json family[H2].p_holm"),
    ("hTwoSentenceBound", "the registered H2 sentence with its p as a bound (DEV-13)", h2_sentence_bound,
     f"{A}/confirmatory.json claims.H2 via make_numbers.SENTENCES hTwoSentence"),
    ("hTwoMDERealized", "H2 MDE at the realized SE (registered formula z_holm x SE; DEV-13)", realized_mde("h2"),
     f"{A}/h2.json confirmatory.se; results/prereg/power_prereg.json conventions.z_holm"),
    ("hThreeMDERealized", "H3 MDE at the realized SE (DEV-13)", realized_mde("h3"),
     f"{A}/h3.json confirmatory.se; results/prereg/power_prereg.json conventions.z_holm"),
    ("mdePowerPct", "power behind every registered MDE (%)", power_pct,
     "results/prereg/power_prereg.json conventions.z_holm; family alpha"),
    ("hThreeICPrimary", "rank IC of the full system's pooled logit, learned weeks", val("h3", "ic", "primary", "mean"),
     f"{A}/h3.json ic.primary.mean"),
    ("hThreeICPrimaryCI", "its NW 95% CI", ci("h3", "ic", "primary", "ci"), f"{A}/h3.json ic.primary.ci"),
    ("hThreeICCodeonly", "rank IC of the code-only twin", val("h3", "ic", "codeonly", "mean"),
     f"{A}/h3.json ic.codeonly.mean"),
    ("hThreeICCodeonlyCI", "its NW 95% CI", ci("h3", "ic", "codeonly", "ci"), f"{A}/h3.json ic.codeonly.ci"),
    ("hThreeLossPrimary", "weekly log loss of the full system", val("h3", "dm_logloss", "loss_a", dec=4),
     f"{A}/h3.json dm_logloss.loss_a"),
    ("hThreeLossCodeonly", "weekly log loss of the code-only twin", val("h3", "dm_logloss", "loss_b", dec=4),
     f"{A}/h3.json dm_logloss.loss_b"),
    ("lnTwoVal", "ln 2, the log loss of a constant one-half forecast", ln_two, "math.log(2)"),
    ("hThreeReversed", "H3 descriptive clause when the CI lies below 0 (DEV-13)", h3_reversed,
     f"{A}/confirmatory.json family[H3]"),
    ("hThreeAbstractText", "H3 clause of the abstract (DEV-13 wording)", abstract_h3, f"{A}/confirmatory.json family[H3]"),
    ("hOneAbstractText", "H1 clause of the abstract (DEV-13 wording)", abstract_h1, f"{A}/confirmatory.json family[H1]"),
    ("mOneBRatio", "share of a planted IC signal the frozen learner recovered (pre-commit M1b)",
     rec(MECH, "M1b", "0.1", "ratio", dec=2, window="W_pre_slice"), "results/pre/mechanical_checks.json M1b.0.1.ratio"),
    ("mOneBPlanted", "the planted IC of M1b", rec(MECH, "M1b", "0.1", "ic_th", dec=2, window="W_pre_slice"),
     "results/pre/mechanical_checks.json M1b.0.1.ic_th"),
    ("alphaSystem", "own-factor selection alpha of the system book (%/yr)", pd1_row("sized", "alpha", "annualized"),
     f"{A}/pd1.json rows.sized.alpha.annualized"),
    ("alphaSystemCI", "its 95% CI (%/yr)", pd1_row("sized", "alpha", "ci"), f"{A}/pd1.json rows.sized.alpha.ci"),
    ("alphaCodeonly", "own-factor alpha of the code-only system (%/yr)", pd1_row("codeonly", "alpha", "annualized"),
     f"{A}/pd1.json rows.codeonly.alpha.annualized"),
    ("alphaCodeonlyCI", "its 95% CI (%/yr)", pd1_row("codeonly", "alpha", "ci"), f"{A}/pd1.json rows.codeonly.alpha.ci"),
    ("dsrHeadline", "DSR of the system book over the two registered headline candidates (M = 2)",
     val("pd1", "dsr", "by_book", "sized", "headline_candidates", "dsr", dec=2),
     f"{A}/pd1.json dsr.by_book.sized.headline_candidates.dsr"),
    ("nTrialsRun", "registered trials that ran", val("pd1", "dsr", "trials", "m_present", integer=True),
     f"{A}/pd1.json dsr.trials.m_present"),
    ("randomSharpeMedian", "median Sharpe of the Random-K books", val("pd1", "random20", "sharpe_p50", dec=2),
     f"{A}/pd1.json random20.sharpe_p50"),
    ("dOneMDE", "D1 MDE per three-month window (AUC)", rec(POWER, "baselines_reduced", "PROBE", "D1", "reduced",
                                                          "three_month_auc_mde", dec=2),
     "results/prereg/power_prereg.json baselines_reduced.PROBE.D1.reduced.three_month_auc_mde"),
    ("dOneFalseMove", "simulated false-move rate of the D1 rule (persistent names)",
     rec(SIMS, "d1", "persistent_names", "name_permutation_p_below_0.05", dec=3,
         script="experiments/analysis/sim_null_rates.py"),
     "results/analysis/simulations/null_rates.json d1.persistent_names"),
    ("dOneBootFalseLow", "false-move rate of the descriptive bootstrap interval, iid names",
     rec(SIMS, "d1", "iid", "bootstrap_interval_excludes_0.5", dec=2, script="experiments/analysis/sim_null_rates.py"),
     "results/analysis/simulations/null_rates.json d1.iid"),
    ("dOneBootFalseHigh", "the same, persistent names",
     rec(SIMS, "d1", "persistent_names", "bootstrap_interval_excludes_0.5", dec=2,
         script="experiments/analysis/sim_null_rates.py"),
     "results/analysis/simulations/null_rates.json d1.persistent_names"),
    ("ngMDE", "NG pooled Spearman MDE at 500 units", rec(POWER, "baselines_reduced", "NG_500", "pooled_spearman_mde"),
     "results/prereg/power_prereg.json baselines_reduced.NG_500.pooled_spearman_mde"),
    ("verbMDE", "VERB pooled Spearman MDE at 2,000 units",
     rec(POWER, "baselines_reduced", "VERB", "reduced", "spearman_pooled_mde"),
     "results/prereg/power_prereg.json baselines_reduced.VERB.reduced.spearman_pooled_mde"),
    ("dTwoMDECell", "D2 re-identification MDE per cell", rec(POWER, "baselines_reduced", "PROBE", "D2", "reduced",
                                                            "reid_company_mde_per_cell"),
     "results/prereg/power_prereg.json baselines_reduced.PROBE.D2.reduced.reid_company_mde_per_cell"),
    ("dTwoCompanyFullCI", "Wilson 95% CI of the full-blinding re-identification rate (%)", d2_ci("full"),
     f"{A}/d2.json by_period_level[W_clean, full, reid_company].ci"),
    ("dTwoCondNamesCI", "Wilson 95% CI (%) of accuracy among name-masked probes that named a firm (DEV-13)",
     wilson_cond("names"), f"{A}/d2.json by_period_level[W_clean, names, reid_company]"),
    ("demotionASR", "typed demotion ASR, removal or halving, pooled (%)", demotion("asr"),
     f"{A}/h1.json pooled[demotion, typed].asr"),
    ("demotionCI", "its week-clustered 95% CI (%)", demotion("ci"), f"{A}/h1.json pooled[demotion, typed].ci"),
    ("demotionN", "typed demotion units", demotion("n"), f"{A}/h1.json pooled[demotion, typed].n"),
    ("demotionProtected", "typed demotion ASR, protected stratum (%)", demotion("asr", "protected"),
     f"{A}/h1.json demotion[typed, protected, A8-1].asr"),
    ("demotionProtectedCI", "its 95% CI (%)", demotion("ci", "protected"), f"{A}/h1.json demotion[typed, protected].ci"),
    ("demotionProtectedExcl", "exclusion ASR, protected stratum (%)", demotion("exclusion_asr", "protected"),
     f"{A}/h1.json demotion[typed, protected].exclusion_asr"),
    ("demotionInBand", "typed demotion ASR, in-band held stratum (%)", demotion("asr", "in_band"),
     f"{A}/h1.json demotion[typed, in_band].asr"),
    ("demotionInBandCI", "its 95% CI (%)", demotion("ci", "in_band"), f"{A}/h1.json demotion[typed, in_band].ci"),
    ("demotionAnchorExclOne", "exclusion ASR with a trusted anchor, family A8-1 (%)",
     demotion("exclusion_asr", "anchor", "A8-1"), f"{A}/h1.json demotion[typed, anchor, A8-1].exclusion_asr"),
    ("demotionAnchorExclTwo", "exclusion ASR with a trusted anchor, family A8-2 (%)",
     demotion("exclusion_asr", "anchor", "A8-2"), f"{A}/h1.json demotion[typed, anchor, A8-2].exclusion_asr"),
    ("nodefAone", "ASR without defenses, family A1 (%)", fam_reader("nodef", "A1"), f"{A}/h1.json per_family[nodef, A1]"),
    ("nodefAthree", "ASR without defenses, family A3 (%)", fam_reader("nodef", "A3"),
     f"{A}/h1.json per_family[nodef, A3]"),
    ("nodefAfour", "ASR without defenses, family A4 (%)", fam_reader("nodef", "A4"),
     f"{A}/h1.json per_family[nodef, A4]"),
    ("nodefAfive", "ASR without defenses, family A5 (%)", fam_reader("nodef", "A5"),
     f"{A}/h1.json per_family[nodef, A5]"),
    ("lmAone", "ASR of the dictionary reader, family A1 (%)", fam_reader("LM", "A1"), f"{A}/h1.json per_family[LM, A1]"),
    ("lmAfour", "ASR of the dictionary reader, family A4 (%)", fam_reader("LM", "A4"), f"{A}/h1.json per_family[LM, A4]"),
    ("asrZeroUpper", "exact 95% upper bound (%) of a zero-success typed family (n = 173; DEV-13)", cp_upper_zero("A1"),
     f"{A}/h1.json per_family[typed, A1].n"),
    ("hOneHolisticCP", "exact 95% interval (%) of the holistic reader's pooled ASR (DEV-13)",
     cp_interval(lambda: _holistic_counts()[0], lambda: _holistic_counts()[1]), f"{A}/h1.json pooled[promotion, H]"),
    ("medDelta", "ASR typed - median aggregation (pp)", contrast("R-med", "estimate"),
     f"{A}/h1.json secondary.contrasts[R-med].estimate"),
    ("medCI", "its 95% CI (pp)", contrast("R-med", "ci"), f"{A}/h1.json secondary.contrasts[R-med].ci"),
    ("voteDelta", "ASR typed - majority vote (pp)", contrast("R-vote", "estimate"),
     f"{A}/h1.json secondary.contrasts[R-vote].estimate"),
    ("voteCI", "its 95% CI (pp)", contrast("R-vote", "ci"), f"{A}/h1.json secondary.contrasts[R-vote].ci"),
    ("lmDelta", "ASR typed - dictionary reader (pp)", contrast("LM", "estimate"),
     f"{A}/h1.json secondary.contrasts[LM].estimate"),
    ("lmCI", "its 95% CI (pp)", contrast("LM", "ci"), f"{A}/h1.json secondary.contrasts[LM].ci"),
    ("hspotDelta", "ASR typed - holistic spot reader (pp)", contrast("H-spot", "estimate"),
     f"{A}/h1.json secondary.contrasts[H-spot].estimate"),
    ("hspotCI", "its 95% CI (pp)", contrast("H-spot", "ci"), f"{A}/h1.json secondary.contrasts[H-spot].ci"),
    ("nEntryChecks", "worst-case-reader entry checks (P1)", val("bound", "bg5", "n_entry_checks", integer=True),
     f"{A}/bound.json bg5.n_entry_checks"),
    ("nProtectionChecks", "worst-case-reader protection checks (P2)",
     val("bound", "bg5", "n_protection_checks", integer=True), f"{A}/bound.json bg5.n_protection_checks"),
    ("pTwoProtectedHeld", "protected held names without an anchor under attack",
     val("h1", "tightness", "P2", "n_protected_held_nonanchor", integer=True),
     f"{A}/h1.json tightness.P2.n_protected_held_nonanchor"),
    ("pTwoShareHalved", "share of those halved (%)", val("h1", "tightness", "P2", "share_halved", dec=0, scale=100),
     f"{A}/h1.json tightness.P2.share_halved"),
    ("lopoMin", "PD2 leave-one-program-out: smallest IC difference (full minus without)", lopo("min"),
     f"{A}/pd2.json leave_one_out"),
    ("lopoMinProg", "the program of that difference", lopo("minprog"), f"{A}/pd2.json leave_one_out"),
    ("lopoMax", "PD2 leave-one-program-out: largest IC difference", lopo("max"), f"{A}/pd2.json leave_one_out"),
    ("lopoMaxProg", "the program of that difference", lopo("maxprog"), f"{A}/pd2.json leave_one_out"),
    ("litSearchDate", "date of the last literature search behind the gap sentence", lit_search_date,
     "literature/meta/lit1_search_log.json date_utc"),
    ("wCleanStartProse", "first decision of W_clean, prose date", prose_date("clean_start"),
     "configs/system_v1.yaml windows.W_clean[0]"),
    ("wCleanEndProse", "last decision of W_clean, prose date", prose_date("clean_end"),
     "configs/system_v1.yaml windows.W_clean[1]"),
    ("wReleaseStartProse", "first decision of W_release, prose date", prose_date("release_start"),
     "configs/system_v1.yaml windows.W_release[0]"),
    ("emuCutoffMonth", "documented knowledge cutoff of emu's LLM (month)", emu_cutoff_month,
     "docs/pricing/openai_model_gpt-4.1-mini.html"),
    ("dOneFirstMonth", "first month of the D1 probe schedule", d1_month("first"), f"{A}/d1.json schedule.months[0]"),
    ("dOneLastMonth", "last month of the D1 probe schedule", d1_month("last"), f"{A}/d1.json schedule.months[-1]"),
    ("dOneCentreMonth", "centre month of the D1 rule's window", d1_month("centre"), f"{A}/d1.json settings.center"),
    ("nDOneWindows", "registered three-month D1 windows", d1_month("windows"), f"{A}/d1.json monthly_windows"),
    ("nSecondaryContrasts", "registered secondary contrasts of H1's family", n_secondary,
     f"{A}/h1.json secondary.m_registered"),
    ("nPromotionFamilies", "registered promotion attack families", n_families("promotion"),
     f"{A}/h1.json per_family (kind promotion)"),
    ("nDemotionFamilies", "registered demotion attack families", n_families("demotion"),
     f"{A}/h1.json per_family (kind demotion)"),
]
# Jev slot: empty unless the descriptive Jev-panel output exists (never an error)
_F = f"{JR} forward"
_AL = f"{JR} aligned2000"
_HF = f"{JR} h2frame"
_CO = f"{JR} cost_latency"
_PR = f"{JR} probes"
_JREL = f"{JR} meta.generations.forward.labels"
JEV: list[tuple[str, str, Getter, str]] = [
    # the slot's original names: the FORWARD post-release statistic (a), and the h2frame earnings AUC (c)
    ("hTwoSpearmanJev", "JEV SLOT: Jev pooled Spearman of E[effect] with AR[0,+1], forward post-release units",
     jev_val("forward", "jev", "spearman_ar", "rho"), f"{_F}.jev.spearman_ar.rho"),
    ("hTwoCIJev", "JEV SLOT: its week-block 95% CI", jev_ci("forward", "jev", "spearman_ar", "ci_week_block"),
     f"{_F}.jev.spearman_ar.ci_week_block"),
    ("hTwoPlaceboJev", "JEV SLOT: Jev pooled Spearman with placebo AR[-5,-1], same units",
     jev_val("forward", "jev", "spearman_ar_placebo", "rho"), f"{_F}.jev.spearman_ar_placebo.rho"),
    ("hTwoAUCEarningsJev", "JEV SLOT: Jev earnings-concept (Item 2.02) AUC on the registered H2 frame (h2frame)",
     jev_val("h2frame", "jev_sec_8k", "earnings_results_vs_2.02", "auc"), f"{_HF}.jev_sec_8k.earnings_results_vs_2.02.auc"),
    ("nJevUnits", "JEV SLOT: forward post-release units with a realized AR window",
     jev_val("forward", "units_used", integer=True), f"{_F}.units_used"),
    ("jevPanelStatus", "JEV SLOT: status word of the Jev rows", jev_text("status"), f"{JR} meta.label"),
    ("jevPanelTiming", "JEV SLOT: timing of the forward units relative to Jev's release", jev_text("timing"), _JREL),
    ("jevRecallClause", "JEV SLOT: what the timing implies for recall", jev_text("recall"), _JREL),
    # (a) forward: more detail and the emu reference on the identical units (a reference, not a test)
    ("jevFwdCIFisher", "Jev forward Spearman, Fisher-z 95% CI", jev_ci("forward", "jev", "spearman_ar", "ci_fisher"),
     f"{_F}.jev.spearman_ar.ci_fisher"),
    ("jevFwdPlaceboCI", "Jev forward placebo, week-block 95% CI",
     jev_ci("forward", "jev", "spearman_ar_placebo", "ci_week_block"), f"{_F}.jev.spearman_ar_placebo.ci_week_block"),
    ("jevFwdWeeks", "forward weeks", jev_val("forward", "jev", "spearman_ar", "weeks", integer=True),
     f"{_F}.jev.spearman_ar.weeks"),
    ("jevFwdUnitsAll", "forward post-release units asked", jev_val("forward", "units_total", integer=True),
     f"{_F}.units_total"),
    ("jevFwdUnitsDropped", "forward units whose AR window is not realized in the data (point-in-time)",
     jev_val("forward", "units_dropped", "AR window not realized by the last session in the data", integer=True),
     f"{_F}.units_dropped"),
    ("jevFwdSpearmanEmu", "emu reference: pooled Spearman with AR[0,+1] on the identical forward units",
     jev_val("forward", "emu_reference_same_units", "spearman_ar", "rho"),
     f"{_F}.emu_reference_same_units.spearman_ar.rho"),
    ("jevFwdCIEmu", "emu reference: its week-block 95% CI",
     jev_ci("forward", "emu_reference_same_units", "spearman_ar", "ci_week_block"),
     f"{_F}.emu_reference_same_units.spearman_ar.ci_week_block"),
    ("jevFwdPlaceboEmu", "emu reference: placebo on the identical forward units",
     jev_val("forward", "emu_reference_same_units", "spearman_ar_placebo", "rho"),
     f"{_F}.emu_reference_same_units.spearman_ar_placebo.rho"),
    ("jevFwdAgreeEmu", "Spearman of Jev and emu E[effect] on the forward units",
     jev_val("forward", "jev_emu_agreement_spearman_E_effect"), f"{_F}.jev_emu_agreement_spearman_E_effect"),
    # (b) aligned2000 agreement (no outcome read)
    ("nJevAligned", "aligned units (Jev, emu, VERB)", jev_val("aligned2000", "units", integer=True), f"{_AL}.units"),
    ("nJevAlignedNG", "aligned units nested for NG", jev_val("aligned2000", "units_ng", integer=True),
     f"{_AL}.units_ng"),
    ("jevAgreeEmuRho", "Spearman of E[effect], Jev vs emu",
     jev_val("aligned2000", "jev_vs_emu", "spearman_E_effect", "rho"), f"{_AL}.jev_vs_emu.spearman_E_effect.rho"),
    ("jevAgreeEmuCI", "its week-block 95% CI", jev_ci("aligned2000", "jev_vs_emu", "spearman_E_effect", "ci_week_block"),
     f"{_AL}.jev_vs_emu.spearman_E_effect.ci_week_block"),
    ("jevAgreeEmuKappa", "Cohen's kappa of the top event type, Jev vs emu",
     jev_val("aligned2000", "jev_vs_emu", "top_event_type", "kappa", dec=2), f"{_AL}.jev_vs_emu.top_event_type.kappa"),
    ("jevAgreeVerbRho", "Spearman of E[effect], Jev vs VERB",
     jev_val("aligned2000", "jev_vs_verb", "spearman_E_effect", "rho"), f"{_AL}.jev_vs_verb.spearman_E_effect.rho"),
    ("jevAgreeVerbKappa", "Cohen's kappa of the top event type, Jev vs VERB",
     jev_val("aligned2000", "jev_vs_verb", "top_event_type", "kappa", dec=2), f"{_AL}.jev_vs_verb.top_event_type.kappa"),
    ("jevAgreeNGRho", "Spearman of E[effect], Jev vs NG (nested 500)",
     jev_val("aligned2000", "jev_vs_ng", "spearman_E_effect", "rho"), f"{_AL}.jev_vs_ng.spearman_E_effect.rho"),
    ("jevAgreeNGKappa", "Cohen's kappa of the top event type, Jev vs NG",
     jev_val("aligned2000", "jev_vs_ng", "top_event_type", "kappa", dec=2), f"{_AL}.jev_vs_ng.top_event_type.kappa"),
    ("jevAgreeEmuVerbRho", "reference: Spearman of E[effect], emu vs VERB, same units",
     jev_val("aligned2000", "reference", "emu_vs_verb", "spearman_E_effect", "rho"),
     f"{_AL}.reference.emu_vs_verb.spearman_E_effect.rho"),
    ("jevAgreeEmuNGRho", "reference: Spearman of E[effect], emu vs NG, same units",
     jev_val("aligned2000", "reference", "emu_vs_ng", "spearman_E_effect", "rho"),
     f"{_AL}.reference.emu_vs_ng.spearman_E_effect.rho"),
    # (c) h2frame 8-K concept agreement (W_clean; contamination unverified; descriptive)
    ("nJevFrameUnits", "h2frame units with a Jev answer", jev_val("h2frame", "jev_units", integer=True),
     f"{_HF}.jev_units"),
    ("jevAUCEarningsCI", "Jev earnings AUC, week-block 95% CI",
     jev_ci("h2frame", "jev_sec_8k", "earnings_results_vs_2.02", "auc_ci"),
     f"{_HF}.jev_sec_8k.earnings_results_vs_2.02.auc_ci"),
    ("jevAUCDeals", "Jev deal-concept (Items 1.01/2.01) AUC",
     jev_val("h2frame", "jev_sec_8k", "deal_or_investment_vs_1.01/2.01", "auc"),
     f"{_HF}.jev_sec_8k.deal_or_investment_vs_1.01/2.01.auc"),
    ("jevAUCDealsCI", "its week-block 95% CI", jev_ci("h2frame", "jev_sec_8k", "deal_or_investment_vs_1.01/2.01", "auc_ci"),
     f"{_HF}.jev_sec_8k.deal_or_investment_vs_1.01/2.01.auc_ci"),
    ("jevAUCManagement", "Jev management-concept (Item 5.02) AUC",
     jev_val("h2frame", "jev_sec_8k", "management_or_labor_vs_5.02", "auc"),
     f"{_HF}.jev_sec_8k.management_or_labor_vs_5.02.auc"),
    ("jevAUCManagementCI", "its week-block 95% CI",
     jev_ci("h2frame", "jev_sec_8k", "management_or_labor_vs_5.02", "auc_ci"),
     f"{_HF}.jev_sec_8k.management_or_labor_vs_5.02.auc_ci"),
    # (d) cost and latency
    ("jevLedgerUSD", "Jev panel spend booked in the JEV-CF ledger (USD)", jev_val("cost_latency", "ledger", "usd", dec=2),
     f"{_CO}.ledger.usd"),
    ("jevPaidRequests", "paid Jev requests over the panel",
     jev_val("cost_latency", "totals_requests_parquet", "paid", integer=True), f"{_CO}.totals_requests_parquet.paid"),
    ("jevUSDPerKItems", "USD per 1,000 items (five requests each), h2frame",
     jev_val("cost_latency", "by_component", "h2frame", "usd_per_1000_items_paid", dec=2),
     f"{_CO}.by_component.h2frame.usd_per_1000_items_paid"),
    ("jevLatencyMedian", "median latency of a paid request (s), h2frame",
     jev_val("cost_latency", "by_component", "h2frame", "latency_paid_s", "median", dec=2),
     f"{_CO}.by_component.h2frame.latency_paid_s.median"),
    ("jevLatencyPNinetyNine", "99th-percentile latency of a paid request (s), h2frame",
     jev_val("cost_latency", "by_component", "h2frame", "latency_paid_s", "p99", dec=2),
     f"{_CO}.by_component.h2frame.latency_paid_s.p99"),
    # (e) probes: lower bounds only
    ("jevDOneAUC", "Jev D1 recall AUC, window centred on 2024-07 (lower bound)",
     jev_val("probes", "d1", "jev", "centre_window", "auc"), f"{_PR}.d1.jev.centre_window.auc"),
    ("jevDOneAUCCI", "its week-block 95% CI (descriptive)", jev_ci("probes", "d1", "jev", "centre_window", "ci_week_block"),
     f"{_PR}.d1.jev.centre_window.ci_week_block"),
    ("jevDOneAUCPeriod", "Jev D1 recall AUC over the probe period 2024-01..2025-06",
     jev_val("probes", "d1", "jev", "probe_period", "auc"), f"{_PR}.d1.jev.probe_period.auc"),
    ("jevDOneAUCPeriodCI", "its week-block 95% CI (descriptive)",
     jev_ci("probes", "d1", "jev", "probe_period", "ci_week_block"), f"{_PR}.d1.jev.probe_period.ci_week_block"),
    ("jevDTwoCompanyFull", "Jev D2 company re-identification accuracy, full blinding (%; chance 20%)",
     jev_d2("full", "reid_company"), f"{_PR}.d2.jev.pooled_by_level[full, reid_company].accuracy"),
    ("jevDTwoYearFull", "Jev D2 year re-identification accuracy, full blinding (%; chance 12.5%)",
     jev_d2("full", "reid_year"), f"{_PR}.d2.jev.pooled_by_level[full, reid_year].accuracy"),
]


def defined_in_numbers() -> dict[str, bool]:
    """name -> True if numbers.tex defines it with a placeholder."""
    out = {}
    if NUMBERS.is_file():
        for m in re.finditer(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", NUMBERS.read_text(), flags=re.MULTILINE):
            out[m.group(1)] = "\\NumPlaceholder{" in m.group(2)
    return out


def render() -> tuple[str, list[str]]:
    existing = defined_in_numbers()
    lines = ["% paper/generated/numbers_extra.tex -- GENERATED by experiments/report/make_numbers_extra.py (DEV-12).",
             "% Registered-output macros of the validity-first text (outline_v2 §5.2); input after generated/numbers.",
             "% Do not edit by hand (CLAUDE.md rule 2).",
             "\\providecommand{\\NumPlaceholder}[1]{\\textbf{[#1]}}", ""]
    missing, clash = [], []
    for name, desc, get, src in MACROS:
        v = get()
        if v is None:
            missing.append(name)
            v = f"\\NumPlaceholder{{{name}}}"
        if name in existing and not existing[name]:
            clash.append(name)  # make_numbers.py already fills it: never overridden here
            continue
        cmd = "\\renewcommand" if name in existing else "\\newcommand"
        lines += [f"% {desc} | source: {src}", f"{cmd}{{\\{name}}}{{{v}}}"]
    lines += ["", "% ===== JEV SLOT: every macro below is empty (and \\IfJevResults takes its second argument) unless",
              "% the descriptive Jev-panel output results/analysis/jev_rows.json exists (experiments/report/_validity.py).",
              f"\\newcommand{{\\IfJevResults}}[2]{{{jev_flag()}}}"]
    for name, desc, get, src in JEV:
        v = get()
        if v is None:
            missing.append(name)
            v = f"\\NumPlaceholder{{{name}}}"
        lines += [f"% {desc} | source: {src}", f"\\newcommand{{\\{name}}}{{{v}}}"]
    if clash:
        raise SystemExit("refused: make_numbers.py already fills " + ", ".join(clash))
    return "\n".join(lines) + "\n", missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is stale or a source is missing")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    text, missing = render()
    if a.list:
        print(text)
    if a.check:
        stale = not OUT.is_file() or OUT.read_text() != text
        print("numbers_extra: " + ("stale" if stale else "up to date") +
              (f"; missing sources: {', '.join(missing)}" if missing else ""))
        return 1 if stale or missing else 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(MACROS)} macros + {len(JEV)} Jev slots)")
    if missing:
        print("MISSING sources (placeholders written): " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
