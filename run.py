"""run.py - 진입점.

  python run.py --config configs/mimir.toml --stage pair
  python run.py --config configs/zerofu.toml --stage pair --set data.zeta=0.1 --set pair.retained=4 --set pair.forget=5
  python run.py --config configs/mimir.toml --stage sequential
  python run.py --config configs/mimir.toml --stage pair --smoke      # 축소 설정으로 1~2분
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from zfu.experiment import Experiment

ROOT = Path(__file__).resolve().parent


def set_key(cfg: dict, dotted: str):
    key, val = dotted.split("=", 1)
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = tomllib.loads(f"v = {val}")["v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="pair", choices=["pretrain", "pair", "sequential"])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text(encoding="utf-8"))
    for s in a.set:
        set_key(cfg, s)
    if a.tag:
        cfg["tag"] = a.tag
    if a.smoke:
        cfg["tag"] = cfg["tag"] + "_smoke"
        cfg["data"]["subset"] = 6000
        cfg["data"]["min_size"] = 5
        cfg.setdefault("fl", {}).update(rounds=2, eval_every=1)
        cfg.setdefault("unlearn", {}).update(rounds=20, log_every=10)
    exp = Experiment(cfg, ROOT)
    if a.stage == "pretrain":
        exp.pretrain()
    elif a.stage == "pair":
        exp.unlearn_pair()
    else:
        exp.sequential()


if __name__ == "__main__":
    main()
