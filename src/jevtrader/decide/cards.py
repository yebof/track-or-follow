"""Decision cards (design §10.1) and the rendered typed reasoning trace (§10.2).

A card is produced by a template from logged values only; no LLM writes it. Two ledgers on every card are exact:
- logit ledger: sum over agents of c_a == L, and L^T + L^W + text-cap line == L (tested to 1e-9);
- weight ledger: the D-sized line items sum to the final weight. Reference-anchored sizing (design_v2_delta BG-2)
  logs "trusted target tau -> reference target -> text multiplier -> turnover -> caps -> final"; entrants show their
  share of the displaced reference weight; V-size-v1 keeps the Day-1 lines (trusted weight, name cap, the binding
  multiplier with its cause, industry cap, scale-down, turnover, post-turnover cap, forced exit).
Every line carries a stable rule id (RK-1, decide.risk.RULE_IDS) and the card states whether a trusted anchor
(BG-3) was present. Evidence, provenance and System-2 blocks come from Layer A's tables and are passed through.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from jevtrader.data.blinding import pseudonym
from jevtrader.decide.risk import MULTIPLIER_CHANNEL, RULE_IDS, Weights
from jevtrader.decide.select import Selection
from jevtrader.learn.features import CATALOG

EPS_W = 1e-12




def _action(prev: float, new: float) -> str:
    if prev <= EPS_W and new <= EPS_W:
        return "none"
    if prev <= EPS_W:
        return "enter"
    if new <= EPS_W:
        return "exit"
    if abs(new - prev) <= 1e-6:
        return "hold"
    return "increase" if new > prev else "decrease"


def build_cards(
    t, scores, sel_ew: Selection | None, w_ew: pd.Series, sel_sz: Selection | None, sized: Weights | None,
    prev_ew: pd.Series | None = None, prev_sz: pd.Series | None = None, *, K: int = 20, J: float = 10,
    evidence: Mapping[str, dict] | None = None, provenance: Mapping[str, dict] | None = None,
    s2: Mapping[str, dict] | None = None, top_n: int = 5, catalog: Mapping = CATALOG,
    anchors: Mapping[str, float] | pd.Series | None = None,
) -> list[dict]:
    """One card per name that is held, entered, exited, or inside the band (R^T <= K + J)."""
    fr = scores.frame
    prev_ew = pd.Series(dtype=float) if prev_ew is None else prev_ew
    prev_sz = pd.Series(dtype=float) if prev_sz is None else prev_sz
    w_sz = sized.final if sized is not None else w_ew
    names = set(fr.index[fr["R_T"] <= K + J]) | set(w_ew.index) | set(w_sz.index)
    names |= set(prev_ew.index[prev_ew > EPS_W]) | set(prev_sz.index[prev_sz > EPS_W])
    sel = sel_sz if sel_sz is not None else sel_ew
    band = sel.band.to_dict() if sel is not None else {}
    frd = fr.to_dict("index")
    agd = scores.agents.to_dict("index")
    qcols = list(scores.questions.columns)
    qpos = {s: k for k, s in enumerate(scores.questions.index)}
    qv, xv = scores.questions.to_numpy(dtype=float), scores.x.reindex(columns=qcols).to_numpy(dtype=float)
    d_ew, d_sz, d_pew, d_psz = (ser.to_dict() for ser in (w_ew, w_sz, prev_ew, prev_sz))
    u = sized.trusted.to_dict() if sized is not None else {}
    tau = sized.tau.to_dict() if sized is not None else {}
    anc = dict(anchors) if anchors is not None else {}
    cards = []
    for s in sorted(names, key=lambda x: (frd.get(x, {}).get("R", 10**9), x)):
        row = frd.get(s)
        card: dict = {"t": pd.Timestamp(t).date().isoformat(), "symbol": s, "snapshot_id": scores.snapshot_id}
        card["decision"] = {
            "action": _action(d_psz.get(s, 0.0), d_sz.get(s, 0.0)), "action_sel_ew": _action(d_pew.get(s, 0.0), d_ew.get(s, 0.0)),
            "w_sel_ew": float(d_ew.get(s, 0.0)), "w_sized": float(d_sz.get(s, 0.0)),
            "w_prev_sized": float(d_psz.get(s, 0.0)), "w_prev_sel_ew": float(d_pew.get(s, 0.0)),
            "rank_L": int(row["R"]) if row is not None else None, "rank_T": int(row["R_T"]) if row is not None else None,
            "band": str(band.get(s, "outside_universe")), "P": float(row["P"]) if row is not None else None,
            "cold_start": bool(scores.cold),
        }
        if row is not None:
            q, x = qv[qpos[s]], xv[qpos[s]]
            top = np.argsort(-np.abs(q), kind="mergesort")[:top_n]
            card["logit"] = {
                "L_T": float(row["L_T"]), "L_W": float(row["L_W"]), "L_W_bar": float(row["L_W_bar"]),
                "cap_line": float(row["cap_line"]), "rho": float(row["rho"]), "kappa": float(scores.kappa),
                "L": float(row["L"]),
                "agents": {a: float(v) for a, v in agd[s].items()},
                "top_questions": [
                    {"feature": qcols[k], "agent": catalog[qcols[k]].agent, "channel": catalog[qcols[k]].channel,
                     "x": float(x[k]), "c": float(q[k])}
                    for k in top if q[k] != 0.0
                ],
            }
        if sized is not None:
            lines = sized.ledger.get(s, [("not_selected", 0.0)])
            card["weights"] = {
                "sizing": sized.sizing, "trusted_weight": float(u.get(s, 0.0)), "trusted_target": float(tau.get(s, 0.0)),
                "lines": [[c, float(v)] for c, v in lines],
                "final": float(d_sz.get(s, 0.0)), "phi_turnover": float(sized.phi),
                "multipliers": {c: {"value": v, "channel": MULTIPLIER_CHANNEL.get(c, "?"), "rule": RULE_IDS.get(c, "?")}
                                for c, v in sized.multipliers.get(s, {}).items()},
                "rules_fired": sorted({RULE_IDS.get(c.split(":", 1)[-1] if c.startswith("mult:") else c, c)
                                       for c, _ in lines[1:]}),
                "anchor": bool(anc.get(s, 0.0) > 0), "entrant": s in set(sized.entrants),
                "displaced": s in set(sized.displaced), "excluded": s in set(sized.excluded),
            }
        for key, src in (("evidence", evidence), ("provenance", provenance), ("s2", s2)):
            if src is not None and s in src:
                card[key] = src[s]
        cards.append(card)
    return cards


def check_card(card: dict, tol: float = 1e-9) -> None:
    """Raise AssertionError unless both ledgers of the card are exact."""
    lg = card.get("logit")
    if lg is not None:
        assert abs(sum(lg["agents"].values()) - lg["L"]) <= tol, "agent contributions do not sum to L"
        assert abs(lg["L_T"] + lg["L_W"] + lg["cap_line"] - lg["L"]) <= tol, "logit ledger does not sum to L"
        assert abs(lg["L_T"] + lg["L_W_bar"] - lg["L"]) <= tol
    wl = card.get("weights")
    if wl is not None:
        assert abs(sum(v for _, v in wl["lines"]) - wl["final"]) <= tol, "weight ledger does not sum to w_final"


# ------------------------------------------------------------------------------------ rendering
def _fmt(x, nd=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def _risk_line(card: dict) -> str:
    wl = card.get("weights")
    if not wl:
        return "not sized (signal book only)"
    parts = []
    for cause, v in wl["lines"][1:]:
        if cause.startswith("mult:"):
            c = cause[5:]
            parts.append(f"{c} x-cut {v:+.4f} ({MULTIPLIER_CHANNEL.get(c, '?')} channel)")
        else:
            parts.append(f"{cause} {v:+.4f}")
    base = wl["lines"][0]
    applied = ", ".join(f"{c} x{m['value']:g} ({m['channel']})" for c, m in wl.get("multipliers", {}).items())
    return (f"{base[0]} {base[1]:.4f}; " + ("; ".join(parts) if parts else "no multiplier, cap or turnover change")
            + (f"; multipliers applied (min binds): {applied}" if applied else ""))


def render(card: dict) -> str:
    """Numbered typed reasoning trace: screening -> events -> memory -> synthesis -> other agents -> contributions -> risk."""
    ev = card.get("evidence", {}) or {}
    d = card["decision"]
    name = ev.get("code_name") or pseudonym(card["symbol"])
    w = d["w_sized"]
    head = (f"{name}, review week. Decision: {d['action']} {w:.2%} (combined rank {d['rank_L']}, trusted rank "
            f"{d['rank_T']}, band: {d['band']}).")
    lines = [head]
    sc = ev.get("screening")
    if sc:
        lines.append(
            f"1. News screening: {sc.get('n_items', 0)} items; {sc.get('n_about', 0)} about the company; "
            f"{sc.get('n_quarantined', 0)} quarantined (instructs_reader p={sc.get('max_instructs_reader', 0):.2f}); "
            f"{sc.get('n_sentences_removed', 0)} sentences removed."
        )
    else:
        lines.append("1. News screening: no news evidence logged for this stock-week.")
    clusters = ev.get("clusters") or []
    if clusters:
        ev_txt = []
        for c in clusters:
            ev_txt.append(
                f"{c.get('event_type', '?')} (p={c.get('p_type', float('nan')):.2f}), effect \"{c.get('effect_level', '?')}\" "
                f"(E={c.get('effect_E', float('nan')):.2f}), important p={c.get('p_important', float('nan')):.2f}, "
                f"{'confirmed' if c.get('p_unconfirmed', 1.0) < 0.5 else 'unconfirmed'}, {c.get('width', 1)} outlets; "
                f"new vs memory (max same p={c.get('max_same_p', 0.0):.2f})"
            )
        lines.append("2. Events: " + " | ".join(ev_txt) + ".")
    else:
        lines.append("2. Events: no kept event cluster.")
    syn = ev.get("synthesis")
    lines.append(
        f"3. Synthesis: net effect \"{syn.get('net_effect', '?')}\"; reaction {syn.get('reaction', '?')} "
        f"(p={syn.get('p_reaction', float('nan')):.2f})." if syn else "3. Synthesis: not asked (no kept cluster)."
    )
    ta = ev.get("trusted_answers")
    if ta:
        lines.append(
            f"4. Trusted agents: trend \"{ta.get('trend', '?')}\"; growth \"{ta.get('growth', '?')}\"; "
            f"valuation {ta.get('valuation', '?')}; industry tailwind {_fmt(ta.get('industry_tailwind'))}."
        )
    else:
        tq = {q["feature"]: q["x"] for q in card.get("logit", {}).get("top_questions", []) if q["channel"] == "T"}
        lines.append("4. Trusted agents: " + (", ".join(f"{k} x~={v:+.2f}" for k, v in tq.items()) or "no large trusted inputs") + ".")
    lg = card.get("logit")
    if lg:
        ag = lg["agents"]
        tr = ", ".join(f"{a} {_fmt(v)}" for a, v in ag.items() if CATALOG_AGENT_CHANNEL.get(a) == "T")
        tx = ", ".join(f"{a} {_fmt(v)}" for a, v in ag.items() if CATALOG_AGENT_CHANNEL.get(a) == "W")
        lines.append(f"5. Contributions to the logit: {tr or 'none'} | {tx or 'none'} (text cap {_fmt(lg['cap_line'])}); "
                     f"L={_fmt(lg['L'])}, P={d['P']:.3f}.")
    lines.append(f"6. Risk: {_risk_line(card)}.")
    if card.get("s2"):
        s2 = card["s2"]
        lines.append(f"7. System-2: gate {s2.get('gate_reason', '?')}; action {s2.get('action', '?')}; "
                     f"{len(s2.get('claims', []))} claims.")
    return "\n".join(lines)


CATALOG_AGENT_CHANNEL = {f.agent: f.channel for f in CATALOG.values()}
