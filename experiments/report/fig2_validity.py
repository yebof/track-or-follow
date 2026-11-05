#!/usr/bin/env python3
"""Figure 2 of the validity-first paper: validity of typed item judgments (outline_v2 §3 "Fig. 2"; style §4).

    .venv/bin/python experiments/report/fig2_validity.py   # -> paper/figures/fig2_validity.{pdf,png} + _gray.png

Marker encoding (round-2 figure review F2-1): an OPEN marker always means a placebo window; every other row is filled.
(a) Registered estimates: the Spearman correlation of an item's expected effect level E[effect] with the focal firm's
    announcement-window abnormal return AR[0,+1], 95% intervals. Weekly means of the confirmatory reader (emu; the
    confirmatory estimate in indigo, its registered controls and sensitivities in light indigo) and the placebo window
    (grey, open); then the pooled registered estimates of VERB (triangle) with its placebo (open) and emu on the same
    units (light indigo).
(b) Exploratory, not pre-registered (grey panel): the statistic by publication time (AR[0,+1]); from the first trade
    after publication (E6), for emu, emu without price reports, the two reference readers (FinBERT, dictionary) and the
    paired weekly difference emu - FinBERT (E8); on AR[0,+1] without and with items whose text reports the price move
    (E5 regex of record); and the reference readers on AR[0,+1] with FinBERT's placebo and the paired difference (E7,
    E8). The dashed line is the registered estimate of (a); the price-report share is printed in the panel.
(c) Registered: agreement of the event-type judgments with the firm's own 8-K item codes (AUC, 95% interval; the
    number of positives at the right), three concepts x readers.

Inputs: results/analysis/h2.json (registered, lock open; refused otherwise) for (a) and (c);
results/exploratory/e5_construct_checks.json, e6_publication_timing.json, e7_reference_readers.json and
e8_timing_robustness.json (exploratory loader; refused unless `exploratory: true`) for (b). The
Jev slot (experiments/report/_validity.py) adds a Jev row to (a) and (c) only if a registered Jev row exists. Every
number drawn or printed comes from these files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _validity as V  # noqa: E402
import style  # noqa: E402

NAME = "fig2_validity"
C = style.COLORS
IND, GREY = C["indigo"], C["grey"]
LIGHT_IND = "#8B84C4"  # indigo at ~55% towards white: "emu on the same units"
XLIM_R = (-0.06, 0.44)
REF_MS = style.MARKER_SIZE - 0.3  # reference readers: slightly smaller marks (F2-2)
XLIM_AUC = (0.28, 1.0)


def _ci(d, key_est="mean", key_ci="ci"):
    return V.get(d, key_est), (V.get(d, key_ci) or [np.nan, np.nan])


def draw_rows(ax, rows, xlim, ref=None):
    """rows: dicts with label, est, ci, color, marker, filled, gap (vertical space above the row), header."""
    y, ys, labels, bold = 0.0, [], [], []
    for r in rows:
        y -= r.get("gap", 1.0)
        ys.append(y)
        labels.append(r["label"])
        bold.append(r.get("header", False))
        if r.get("header") or r.get("est") is None or not np.isfinite(r["est"]):
            continue
        lo, hi = r["ci"]
        ax.plot([lo, hi], [y, y], color=r["color"], lw=style.LW_ERR * 1.4, solid_capstyle="butt", zorder=2)
        ms = r.get("ms", style.MARKER_SIZE + (0.4 if r["marker"] in ("^", "h") else 0))
        ax.plot([r["est"]], [y], marker=r["marker"], ms=ms,
                mfc=r["color"] if r.get("filled", True) else "white", mec=r["color"], mew=0.8, ls="none", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for t, b in zip(ax.get_yticklabels(), bold):
        if b:
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(min(ys) - 0.7, 0.2)
    ax.set_xlim(*xlim)
    ax.spines["left"].set_visible(False)
    return ys


def panel_a(ax, h2):
    impl = h2["per_implementation"]
    st = h2.get("settings") or {}
    w0, w1 = st["ar_window"], st["placebo_window"]
    drop = next(k for k in h2["sensitivities"] if k.startswith("drop_first_"))
    placebo = f"placebo AR[\N{MINUS SIGN}{abs(w1[0])}, \N{MINUS SIGN}{abs(w1[1])}]"
    em = V.READERS["emu"]
    rows = [{"label": f"weekly mean, emu ({h2['confirmatory']['n_weeks']} weeks)", "header": True, "gap": 0.6},
            {"label": "confirmatory estimate", "est": h2["confirmatory"]["mean"], "ci": h2["confirmatory"]["ci"],
             "color": IND, "marker": "o"},
            {"label": f"after release ({h2['sensitivities']['W_release']['n_weeks']} weeks)",
             "est": h2["sensitivities"]["W_release"]["mean"], "ci": h2["sensitivities"]["W_release"]["ci"],
             "color": LIGHT_IND, "marker": "o"},
            {"label": f"published before {st['pre_open_before']} ET",
             "est": h2["controls"]["pre_open_only"]["mean"], "ci": h2["controls"]["pre_open_only"]["ci"],
             "color": LIGHT_IND, "marker": "o"},
            {"label": "prior looks excluded", "est": h2["controls"]["prior_look_excluded"]["mean"],
             "ci": h2["controls"]["prior_look_excluded"]["ci"], "color": LIGHT_IND, "marker": "o"},
            {"label": f"first {drop.rsplit('_', 1)[-1]} weeks dropped", "est": h2["sensitivities"][drop]["mean"],
             "ci": h2["sensitivities"][drop]["ci"], "color": LIGHT_IND, "marker": "o"},
            {"label": placebo, "est": h2["placebo_ar_m5_m1"]["mean"], "ci": h2["placebo_ar_m5_m1"]["ci"],
             "color": GREY, "marker": "o", "filled": False},
            {"label": "pooled over items", "header": True, "gap": 1.3},
            {"label": f"{em[0]} (all {h2['confirmatory']['n_units']:,} items)",
             "est": impl["emu"]["spearman_ar_pooled"]["rho"], "ci": impl["emu"]["spearman_ar_pooled"]["ci_week_block"],
             "color": IND, "marker": "o"}]
    jev = V.jev_key(h2)
    for r in ["VERB"] + ([jev] if jev else []):  # NG's rows are in App. C (Table tab:ng; PC review M3)
        d = impl.get(r) or {}
        if d.get("status") != "ok":
            continue
        lab, col, mk = V.READERS.get(r, V.READERS["Jev"])
        rows += [{"label": f"{lab} ({d['n_units']:,} items)", "est": d["spearman_ar"]["rho"],
                  "ci": d["spearman_ar"]["ci_week_block"], "color": col, "marker": mk, "gap": 1.0},
                 {"label": f"{lab} placebo", "est": d["spearman_ar_placebo"]["rho"],
                  "ci": d["spearman_ar_placebo"]["ci_week_block"], "color": col, "marker": mk, "filled": False,
                  "gap": 0.85},
                 {"label": f"emu on {lab}'s items", "est": d["emu_same_units"]["rho"],
                  "ci": d["emu_same_units"]["ci_week_block"], "color": LIGHT_IND, "marker": "o", "gap": 0.85}]
    ys = draw_rows(ax, rows, XLIM_R)
    ax.axvline(0, color=GREY, lw=style.LW_AXES, zorder=1)
    ax.set_xlabel(f"Spearman correlation with AR[{w0[0]}, +{w0[1]}]")
    return ys


def panel_b(ax, e5, e6, e7, e8, h2):
    """Exploratory construct checks (grey panel): timing, the publication-aligned criterion with the reference readers
    and the paired emu - FinBERT difference, price reports, and the reference readers on AR[0,+1] (E5-E8). Group
    headers name the return window (F2-4); every row is filled except FinBERT's placebo (open = placebo, F2-1)."""
    ax.set_facecolor(V.PANEL_X_BG)
    w, d = e5["weekly"], e5["dictionary_reference"]
    tm, pa = e6["timing"], e6["publication_aligned"]
    rd = V.get(e8, "readers", "per_reader") or {}
    pdf = V.get(e8, "readers", "paired_differences", "emu_minus_finbert") or {}
    g, fb_m, dc_m = V.DICT_COLOR, V.FINBERT_MARKER, V.DICT_MARKER
    rows = [{"label": "by publication time; AR[0, +1]", "header": True, "gap": 0.6}]
    for key, lab in (("pre_open", "before the open"), ("intraday", "during the session"),
                     ("after_close_or_nonsession", "after a close or outside a session")):
        t = tm[key]
        rows.append({"label": f"{lab} ({100 * t['share']:.0f}%)", "est": t["ar_0_1"]["mean"], "ci": t["ar_0_1"]["ci"],
                     "color": IND, "marker": "o", "gap": 0.9})
    rows += [{"label": "first trade after publication to next close", "header": True, "gap": 1.25},
             {"label": "emu, all items", "est": pa["all"]["mean"], "ci": pa["all"]["ci"], "color": IND,
              "marker": "o", "gap": 0.9},
             {"label": "emu, no price report", "est": pa["no_price"]["mean"], "ci": pa["no_price"]["ci"],
              "color": LIGHT_IND, "marker": "o", "gap": 0.9}]
    for key, lab, mk in (("finbert", "FinBERT tone", fb_m), ("dictionary", "dictionary tone", dc_m)):
        x = V.get(rd, key, "ar_pub")
        if x:
            rows.append({"label": lab, "est": x["mean"], "ci": x["ci"], "color": g, "marker": mk, "ms": REF_MS,
                         "gap": 0.9})
    if pdf.get("ar_pub"):
        rows.append({"label": "emu \N{MINUS SIGN} FinBERT, paired", "est": pdf["ar_pub"]["mean"],
                     "ci": pdf["ar_pub"]["ci"], "color": C["dark_grey"], "marker": "s", "ms": REF_MS - 0.4,
                     "gap": 0.9})
    rows += [{"label": "price reports (fixed pattern); AR[0, +1]", "header": True, "gap": 1.25},
             {"label": "no price report", "est": w["no_price"]["mean"], "ci": w["no_price"]["ci"], "color": IND,
              "marker": "o", "gap": 0.9},
             {"label": "items that report the move", "est": w["price_only"]["mean"], "ci": w["price_only"]["ci"],
              "color": IND, "marker": "o", "gap": 0.9}]
    s = e5.get("either_classifier_sensitivity") or {}
    if "no_price_either" in s:
        rows.append({"label": "neither classifier flags", "est": s["no_price_either"]["mean"],
                     "ci": s["no_price_either"]["ci"], "color": LIGHT_IND, "marker": "o", "gap": 0.9})
    rows.append({"label": "reference readers; AR[0, +1]", "header": True, "gap": 1.25})
    fb = (e7 or {}).get("finbert") or {}
    if "all" in fb:
        rows.append({"label": "FinBERT tone", "est": fb["all"]["mean"], "ci": fb["all"]["ci"], "color": g,
                     "marker": fb_m, "ms": REF_MS, "gap": 0.9})
    if V.get(rd, "finbert", "placebo"):
        rows.append({"label": "FinBERT placebo", "est": rd["finbert"]["placebo"]["mean"],
                     "ci": rd["finbert"]["placebo"]["ci"], "color": g, "marker": fb_m, "ms": REF_MS,
                     "filled": False, "gap": 0.9})
    rows.append({"label": "dictionary tone", "est": d["all"]["mean"], "ci": d["all"]["ci"], "color": g,
                 "marker": dc_m, "ms": REF_MS, "gap": 0.9})
    if pdf.get("ar_0_1"):
        rows.append({"label": "emu \N{MINUS SIGN} FinBERT, paired", "est": pdf["ar_0_1"]["mean"],
                     "ci": pdf["ar_0_1"]["ci"], "color": C["dark_grey"], "marker": "s", "ms": REF_MS - 0.4,
                     "gap": 0.9})
    ys = draw_rows(ax, rows, XLIM_R)
    ref = h2["confirmatory"]["mean"]
    ax.axvline(ref, color=IND, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0, color=GREY, lw=style.LW_AXES, zorder=1)
    ax.text(ref + 0.008, 0.02, "registered\nestimate (a)", transform=ax.get_xaxis_transform(), fontsize=style.FS_TICK,
            color=IND, va="bottom", ha="left", linespacing=0.95)
    share = e5["price_report"]["share_units"]
    # the coder the paper macros print (spec audit.human_codes: the human coder once it exists)
    prev = V.get(e5, "audit", V.get(e5, "audit", "paper_coder") or "model_coder", "coder_prevalence")
    txt = f"pattern flags {100 * share:.1f}%" + (f"; audit \u2248{100 * prev:.1f}%" if prev else "")
    yb = ys[[r.get("label") for r in rows].index("price reports (fixed pattern); AR[0, +1]")]
    ax.text(0.985, yb, txt, transform=ax.get_yaxis_transform(), fontsize=style.FS_TICK, color=C["dark_grey"],
            ha="right", va="center", zorder=4, bbox=dict(boxstyle="square,pad=0.15", fc=V.PANEL_X_BG, ec="none"))
    ax.set_xlabel("Spearman correlation with the group's return window")
    return ys


def panel_c(ax, h2):
    impl = h2["per_implementation"]
    jev = V.jev_key(h2)
    readers = [r for r in ["emu", "VERB"] + ([jev] if jev else []) if V.get(impl, r, "status") == "ok"]
    rows, pos = [], []
    for key, label, codes in V.SEC_CONCEPTS:
        short = {"Earnings results": "earnings", "Deal or investment": "deal", "Management or labor": "management"}
        rows.append({"label": f"{short.get(label, label.lower())} vs Item{'s' if '/' in codes else ''} {codes}",
                     "header": True, "gap": 1.0 if rows else 0.6})
        for r in readers:
            s = impl["emu"]["sec_8k_full_frame"][key] if r == "emu" else impl[r]["sec_8k"][key]
            lab, col, mk = V.READERS.get(r, V.READERS["Jev"])
            rows.append({"label": lab, "est": s["auc"], "ci": s["auc_ci"], "color": col, "marker": mk, "gap": 0.8})
            pos.append(int(round(s["prevalence"] * s["n"])))
    ys = draw_rows(ax, rows, XLIM_AUC)
    ax.axvline(0.5, color=GREY, lw=style.LW_AXES, ls=(0, (3, 2)), zorder=1)
    k = 0
    for r, y in zip(rows, ys):
        if r.get("header"):
            continue
        ax.text(1.0, y, f"{pos[k]:,}", transform=ax.get_yaxis_transform(), fontsize=style.FS_TICK,
                color=C["dark_grey"], ha="right", va="center")
        k += 1
    ax.text(0.51, ys[0], "chance", fontsize=style.FS_TICK, color=GREY, ha="left", va="center")
    ax.text(1.0, 0.2, "positives", transform=ax.get_yaxis_transform(), fontsize=style.FS_TICK,
            color=C["dark_grey"], ha="right", va="bottom")
    ax.set_xlabel("AUC vs the firm's 8-K item")
    ax.set_xticks([0.4, 0.6, 0.8])
    return ys


def label_panels(fig, specs, y_pad=0.03, italic_below=False):
    """Bold lower-case panel letter at the left edge of each panel's tick labels, the panel title after it, and an
    italic 6.5 pt status label for an exploratory panel (outline_v2 §4): after the title, or on a second line under
    it when `italic_below` (narrow panels)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter, title, italic in specs:
        bb = ax.get_tightbbox(r).transformed(inv)
        top = ax.get_position().y1 + y_pad
        base = top + (style.FS_GLOSS * 1.35 / 72 / fig.get_figheight() if italic_below else 0.0)
        t = fig.text(bb.x0, base, letter, fontsize=style.FS_PANEL, fontweight="bold", ha="left", va="baseline")
        x = t.get_window_extent(r).transformed(inv).x1 + 0.006
        t2 = fig.text(x, base, title, fontsize=style.FS_LABEL, ha="left", va="baseline")
        if italic:
            if italic_below:
                fig.text(x, top, italic, fontsize=style.FS_GLOSS, fontstyle="italic", ha="left", va="baseline",
                         color=C["dark_grey"])
            else:
                x2 = t2.get_window_extent(r).transformed(inv).x1 + 0.005
                fig.text(x2, base, f"({italic})", fontsize=style.FS_GLOSS, fontstyle="italic", ha="left",
                         va="baseline", color=C["dark_grey"])


def main(argv=None) -> int:
    style.apply()
    h2 = V.require_registered("h2")
    e5 = V.exploratory("e5_construct_checks")
    e6 = V.exploratory("e6_publication_timing")
    e7 = V.exploratory("e7_reference_readers")
    e8 = V.exploratory("e8_timing_robustness")
    if e5 is None or e6 is None:
        raise SystemExit("results/exploratory/e5_construct_checks.json or e6_publication_timing.json missing")
    fig = plt.figure(figsize=(style.FULL_WIDTH, 2.72))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.05, 0.85], wspace=1.02, left=0.2, right=0.985, top=0.9,
                  bottom=0.135)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    panel_a(axa, h2)
    panel_b(axb, e5, e6, e7, e8, h2)
    panel_c(axc, h2)
    for ax in (axa, axb):
        ax.set_xticks(np.arange(0.0, XLIM_R[1], 0.1))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "\N{MINUS SIGN}")))
    label_panels(fig, ((axa, "a", "Registered estimates", None),
                       (axb, "b", "Construct checks", "exploratory, not pre-registered"),
                       (axc, "c", "8-K item agreement (registered)", None)))
    style.save(fig, NAME, tight=False)  # exact text width: LaTeX does not rescale the type (F3-1)
    style.check_grayscale(style.FIG_DIR / f"{NAME}.png")
    print(f"{NAME} -> paper/figures/{NAME}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
