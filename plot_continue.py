"""plot_continue.py - 주제 A 결과 그림: 지운 뒤 라운드별로 지운 것이 되살아나는지, 남은 사람이 괜찮은지.

  python plot_continue.py --tag mimir_mnist_z001
  → results/<tag>/fig_continue.png  (왼쪽부터 C_f 데이터 정확도, 백도어 성공률, 남은 클라이언트 평균 정확도)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PANELS = [("df_max", "지울 클라이언트 데이터 정확도 (남은 이름표 중 최대)"),
          ("asr_max", "백도어 성공률 (남은 이름표 중 최대)"),
          ("ret_mean", "남은 클라이언트 평균 정확도")]
STYLE = {"retrain": dict(color="black", ls="--"), "none": dict(color="gray")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    res = ROOT / "results" / a.tag
    with open(res / "continue.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, _ in PANELS:
            if r.get(k):
                series[r["protocol"]][k].append((int(r["t"]), float(r[k])))
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 3.6))
    for ax, (k, title) in zip(axes, PANELS):
        for proto, d in series.items():
            if k in d:
                xs, ys = zip(*sorted(d[k]))
                ax.plot(xs, ys, label=proto, **STYLE.get(proto, {}))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("지운 뒤 연합학습 라운드")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "fig_continue.png", dpi=150)
    print(res / "fig_continue.png")


if __name__ == "__main__":
    main()
