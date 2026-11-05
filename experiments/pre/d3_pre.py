"""D3 pre-period quantities (design_v1 §13.7 D3; evaluation_first D3): blinded vs unblinded typed system on the W_pre
slice x S30.

    .venv/bin/python experiments/pre/d3_pre.py --blinded pre_blinded_emu_wpre_s30 --unblinded pre_unblinded_emu_wpre_s30 \
        --cap-root <scratch>/caproot --work <scratch>/d3 --out results/pre/d3_pre.json

Both generations go through the same Layer B (primary cell, P-clean cold start at the first W_pre decision; prices
capped before W_clean, `caproot`) with identical code features; only the S1 judgments differ. On the S30 names of each
learned week (the names whose judgments the arms produced) it computes:
- the weekly Spearman IC of the pooled logit L with r - m, per arm, and Delta_pre,t = IC_unblinded,t - IC_blinded,t:
  mean, NW(4) t and a moving-block bootstrap 95% CI of the mean (block 4 weeks, 10,000 draws, seeded);
- the same for each agent's contribution c_a (the Layer-B agent logits: technical, fundamentals, macro, news,
  sentinel, policy) and for every raw S1 feature on its covered S30 names (x_mac is a built-in placebo: M's requests
  carry no identity, are byte-identical across arms and share their cached answers, so its Delta must be 0);
- log loss of P against T1 per arm and Delta log loss (DM-style NW(4) test on the weekly loss differences).
The weekly series are written out so that DiD = Delta_pre - Delta_clean can be formed with independent week-block
bootstraps once W_clean is unlocked (Delta_clean needs the S30 W_clean unblinded run). Nothing is registered.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "pre"))

import caproot  # noqa: E402

from jevtrader.eval import metrics as M  # noqa: E402
from jevtrader.learn.features import CATALOG  # noqa: E402
from jevtrader.pipeline import layer_b as LB  # noqa: E402
from jevtrader.pipeline.registry import _plain  # noqa: E402

WINDOW = "W_pre_slice"
SEED = 20261002
N_BOOT = 10_000
BLOCK = 4
EPS = 1e-6


def block_bootstrap_mean(x: pd.Series, block: int = BLOCK, n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    """Circular moving-block bootstrap 95% percentile CI of the mean of a weekly series."""
    v = pd.Series(x).dropna().to_numpy()
    n = len(v)
    if n < 2 * block:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    means = v[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summ(x: pd.Series) -> dict:
    x = pd.Series(x).dropna()
    return {"n_weeks": int(len(x)), "mean": float(x.mean()) if len(x) else float("nan"),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"), "t_nw4": M.t_stat_newey_west(x, 4),
            "ci95_block_bootstrap": block_bootstrap_mean(x)}


def weekly_ic(df: pd.DataFrame, col: str, rel: pd.DataFrame, names) -> pd.Series:
    sc = df.pivot(index="t", columns="symbol", values=col)
    mask = pd.DataFrame(False, index=sc.index, columns=sc.columns)
    for t in sc.index:
        mask.loc[t, [s for s in names.get(t, ()) if s in mask.columns]] = True
    return M.information_coefficient(sc.where(mask), rel)


def arm(gen_id: str, cap: Path, work: Path):
    cfg = LB.DecisionConfig.from_yaml(gen_id=gen_id, name="d3")
    start, end = LB.resolve_window(WINDOW)
    inputs = LB.load_inputs(cfg, start, end, ROOT, data_root=cap)
    assert inputs.prices.sessions.max() <= caproot.PRICE_CAP
    res = LB.evaluate(cfg, WINDOW, inputs, root=work, register=False, write=False)
    return cfg, inputs, res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blinded", required=True)
    ap.add_argument("--unblinded", required=True)
    ap.add_argument("--cap-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    cap = caproot.capped_root(Path(a.cap_root))
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    arms = {"blinded": arm(a.blinded, cap, work), "unblinded": arm(a.unblinded, cap, work)}
    cfg_b, in_b, res_b = arms["blinded"]
    _, in_u, res_u = arms["unblinded"]
    focus = {pd.Timestamp(t): tuple(v) for t, v in (in_b.focus or {}).items()}
    assert focus and {pd.Timestamp(t): tuple(v) for t, v in (in_u.focus or {}).items()} == focus, "focus names differ"
    lab = in_b.labels.table
    rel = lab.assign(rel=lab["r"] - lab["m"]).pivot(index="t", columns="symbol", values="rel")
    pd.testing.assert_frame_equal(in_b.labels.table, in_u.labels.table)
    code = [f for f in cfg_b.allocator.trusted if CATALOG[f].kind == "code"]
    rb, ru = in_b.raw, in_u.raw
    same_code = rb[rb.feature.isin(code)].sort_values(["t", "symbol", "feature"]).reset_index(drop=True).equals(
        ru[ru.feature.isin(code)].sort_values(["t", "symbol", "feature"]).reset_index(drop=True))
    out: dict = {"window": WINDOW, "gens": {"blinded": a.blinded, "unblinded": a.unblinded},
                 "price_cap": str(caproot.PRICE_CAP.date()), "identical_code_features": bool(same_code),
                 "identical_labels": True, "focus_stock_weeks": int(sum(len(v) for v in focus.values())),
                 "cfg_sha": {"blinded": res_b.cfg_sha, "unblinded": res_u.cfg_sha},
                 "inputs_sha": {"blinded": res_b.inputs_sha, "unblinded": res_u.inputs_sha}}
    pb, pu = [r.log.predictions[~r.log.predictions["cold"]] for r in (res_b, res_u)]
    # ---- pooled system
    series = {}
    ic = {}
    for name, p in (("blinded", pb), ("unblinded", pu)):
        ic[name] = weekly_ic(p, "L", rel, focus)
        series[f"ic_L_{name}"] = ic[name]
    d = (ic["unblinded"] - ic["blinded"]).dropna()
    series["delta_ic_L"] = d
    out["pooled"] = {"ic_blinded": summ(ic["blinded"]), "ic_unblinded": summ(ic["unblinded"]), "delta_pre": summ(d)}
    # ---- agents (contribution logits c_a)
    agents = sorted(c[2:] for c in pb.columns if c.startswith("c_"))
    out["agents"] = {}
    for ag in agents:
        icb, icu = weekly_ic(pb, f"c_{ag}", rel, focus), weekly_ic(pu, f"c_{ag}", rel, focus)
        dd = (icu - icb).dropna()
        series[f"delta_ic_c_{ag}"] = dd
        out["agents"][ag] = {"ic_blinded": summ(icb), "ic_unblinded": summ(icu), "delta_pre": summ(dd)}
    # ---- raw S1 features on covered focus names
    s1 = [f for f in (*cfg_b.allocator.trusted, *cfg_b.allocator.untrusted) if CATALOG[f].kind == "s1"]
    out["features"] = {}
    for f in s1:
        vals = {}
        for name, raw in (("blinded", rb), ("unblinded", ru)):
            g = raw[(raw["feature"] == f) & raw["covered"].astype(bool)]
            vals[name] = weekly_ic(g.rename(columns={"value": "v"}), "v", rel, focus)
        dd = (vals["unblinded"] - vals["blinded"]).dropna()
        gb = rb[(rb["feature"] == f)].set_index(["t", "symbol"])["value"]
        gu = ru[(ru["feature"] == f)].set_index(["t", "symbol"])["value"]
        k = gb.index.intersection(gu.index)
        identical = bool(np.allclose(gb.reindex(k).to_numpy(dtype=float), gu.reindex(k).to_numpy(dtype=float),
                                     rtol=0, atol=0, equal_nan=True)) and len(gb) == len(gu)
        out["features"][f] = {"agent": CATALOG[f].agent, "ic_blinded": summ(vals["blinded"]),
                              "ic_unblinded": summ(vals["unblinded"]), "delta_pre": summ(dd),
                              "values_identical_across_arms": identical}
    # ---- log loss of P (T1) on the focus names
    def ll_rows(p):
        j = p.merge(lab[["t", "symbol", "y"]], on=["t", "symbol"]).dropna(subset=["y"])
        return j[[s in set(focus.get(t, ())) for t, s in zip(j["t"], j["symbol"])]]

    jb, ju = ll_rows(pb), ll_rows(pu)
    jj = jb[["t", "symbol", "P", "y"]].merge(ju[["t", "symbol", "P"]], on=["t", "symbol"], suffixes=("_b", "_u"))
    llb = M.log_loss(jj["P_b"].to_numpy(), jj["y"].to_numpy())
    llu = M.log_loss(jj["P_u"].to_numpy(), jj["y"].to_numpy())

    def wk_ll(pcol):
        p = np.clip(jj[pcol].to_numpy(dtype=float), EPS, 1 - EPS)
        y = jj["y"].to_numpy(dtype=float)
        return pd.Series(-(y * np.log(p) + (1 - y) * np.log(1 - p)), index=jj["t"]).groupby(level=0).mean()

    dl = wk_ll("P_u") - wk_ll("P_b")
    series["delta_logloss"] = dl
    out["log_loss"] = {"n": int(len(jj)), "blinded": llb, "unblinded": llu, "delta_pre": llu - llb,
                       "weekly_delta": summ(dl),
                       "dm": M.dm_logloss(jj["P_u"].to_numpy(), jj["P_b"].to_numpy(), jj["y"].to_numpy(),
                                          jj["t"].to_numpy())}
    out["weekly_series"] = {k: {pd.Timestamp(t).date().isoformat(): float(v) for t, v in s.dropna().items()}
                            for k, s in series.items()}
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(_plain(out), indent=1, sort_keys=True, default=str))
    print(json.dumps({k: out[k] for k in ("pooled", "log_loss")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
