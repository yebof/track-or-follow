"""Layer B on the W_pre slice for a PRE generation: `experiments/run_layer_b.py` with the price bars capped before
W_clean (`caproot.capped_root`), so no return after a W_clean decision cutoff is ever computed.

    .venv/bin/python experiments/pre/layer_b_pre.py --cap-root <scratch>/caproot -- --gen-id pre_blinded_emu_wpre_s30 \
        --window W_pre_slice --variants all --composition --bound-check --attribution --sharpe-tests

Everything after `--` goes to run_layer_b.main unchanged (outputs under results/eval/, the trial registry unless
--no-register). The window must be W_pre_slice or lie inside it; anything else is refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "pre"))

import caproot  # noqa: E402
import run_layer_b as RLB  # noqa: E402

from jevtrader.pipeline import layer_b as LB  # noqa: E402

W_PRE = (pd.Timestamp("2023-04-07"), pd.Timestamp("2024-05-31"))


def capped_loader(cap_root: Path):
    orig = LB.load_inputs

    def load(cfg, start, end, root=ROOT, **kw):
        if not (W_PRE[0] <= pd.Timestamp(start) and pd.Timestamp(end) <= W_PRE[1]):
            raise SystemExit(f"layer_b_pre: window {start}..{end} is not inside the W_pre slice")
        inputs = orig(cfg, start, end, root, data_root=cap_root, **kw)
        last = inputs.prices.sessions.max()
        if last > caproot.PRICE_CAP:
            raise SystemExit(f"layer_b_pre: prices reach {last.date()}, past the cap {caproot.PRICE_CAP.date()}")
        return inputs

    return load


def main(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cap-root", required=True)
    a = ap.parse_args(own)
    cap = caproot.capped_root(Path(a.cap_root))
    RLB.load_inputs = capped_loader(cap)
    return RLB.main(rest)


if __name__ == "__main__":
    main()
