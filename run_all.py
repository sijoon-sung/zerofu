"""run_all.py - 한 번에 전부 돌리기 (딸칵).

  python run_all.py                  # 재현 전체: 두 방법 × 4 데이터 × ζ 2개 × (C_r, C_f) 2쌍 = 32 작업 + 요약표
  python run_all.py --estimate       # 이 컴퓨터에서 얼마나 걸릴지 측정만 (1~2분)
  python run_all.py --workers 3      # 작업을 동시에 3개씩 (GPU 메모리 8GB 이상이면 2~3 권장)
  python run_all.py --only mimir --datasets mnist cifar10
  python run_all.py --summary        # 이미 나온 결과로 요약표만 다시

작업 하나 = 원래 모델 학습 → 재학습(C_f 빼고) → 지우기 → 평가. 중간에 끊겨도 다시 실행하면 이어 간다.
결과: results/<tag>/results.csv, 요약: results/summary.md (논문 표 III·표 1 값과 나란히).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

DATASETS = ["mnist", "fmnist", "svhn", "cifar10"]
SIZES = {"mnist": (60000, 1, 28), "fmnist": (60000, 1, 28), "svhn": (73257, 3, 32), "cifar10": (50000, 3, 32)}
PAIRS = {0.01: [(0, 1), (8, 9)], 0.1: [(4, 5), (3, 6)]}           # 두 논문 표의 (C_r, C_f)
METHODS = {"mimir": "configs/mimir.toml", "zerofu": "configs/zerofu.toml"}

# 논문 수치 (%): (Origin Dr, Origin Df, Retrained Dr, Retrained Df, 방법 Dr, 방법 Df)
PAPER = {
    "mimir": {  # Mimir 표 III
        ("mnist", 0.01, 0, 1): (94.76, 93.11, 98.46, 0.04, 87.92, 0.00), ("mnist", 0.01, 8, 9): (93.49, 96.82, 98.52, 0.00, 92.11, 0.00),
        ("mnist", 0.1, 4, 5): (99.32, 99.27, 99.58, 50.81, 94.34, 33.75), ("mnist", 0.1, 3, 6): (97.65, 98.05, 97.29, 25.61, 91.75, 33.01),
        ("svhn", 0.01, 0, 1): (96.06, 95.13, 96.02, 4.87, 96.02, 1.21), ("svhn", 0.01, 8, 9): (96.03, 99.99, 96.17, 0.00, 95.99, 0.00),
        ("svhn", 0.1, 4, 5): (98.14, 79.74, 97.19, 69.70, 98.03, 46.91), ("svhn", 0.1, 3, 6): (85.19, 80.37, 84.40, 54.40, 83.19, 21.93),
        ("fmnist", 0.01, 0, 1): (97.39, 96.60, 99.99, 4.58, 100.0, 9.20), ("fmnist", 0.01, 8, 9): (96.99, 99.13, 99.99, 2.83, 99.05, 5.90),
        ("fmnist", 0.1, 4, 5): (95.17, 94.75, 98.18, 11.70, 93.24, 15.31), ("fmnist", 0.1, 3, 6): (90.06, 98.68, 90.29, 65.04, 77.69, 47.52),
        ("cifar10", 0.01, 0, 1): (98.78, 100.0, 98.78, 0.00, 98.78, 0.00), ("cifar10", 0.01, 8, 9): (81.66, 99.97, 80.65, 0.00, 79.04, 0.00),
        ("cifar10", 0.1, 4, 5): (72.66, 66.71, 81.19, 6.54, 71.01, 7.07), ("cifar10", 0.1, 3, 6): (69.08, 63.36, 91.61, 3.26, 63.22, 2.19),
    },
    "zerofu": {  # ZeroFU 표 1
        ("mnist", 0.01, 0, 1): (96.40, 99.22, 96.30, 0.24, 92.63, 0.00), ("mnist", 0.01, 8, 9): (98.54, 99.91, 98.81, 0.01, 97.05, 0.01),
        ("mnist", 0.1, 4, 5): (99.10, 99.26, 99.03, 57.18, 96.03, 38.95), ("mnist", 0.1, 3, 6): (96.02, 97.29, 94.41, 83.79, 91.04, 83.54),
        ("svhn", 0.01, 0, 1): (96.02, 95.13, 96.02, 0.00, 96.02, 0.00), ("svhn", 0.01, 8, 9): (95.99, 99.99, 95.84, 0.00, 95.99, 0.00),
        ("svhn", 0.1, 4, 5): (98.13, 70.87, 98.13, 60.26, 98.13, 46.60), ("svhn", 0.1, 3, 6): (88.23, 59.93, 86.08, 54.60, 88.26, 45.41),
        ("fmnist", 0.01, 0, 1): (99.45, 99.06, 99.45, 6.27, 99.45, 8.69), ("fmnist", 0.01, 8, 9): (99.98, 99.44, 99.50, 12.96, 99.98, 12.13),
        ("fmnist", 0.1, 4, 5): (83.12, 85.91, 88.13, 13.35, 79.78, 16.11), ("fmnist", 0.1, 3, 6): (76.74, 98.11, 88.41, 13.88, 84.81, 17.31),
        ("cifar10", 0.01, 0, 1): (98.78, 100.0, 98.78, 0.00, 98.78, 0.00), ("cifar10", 0.01, 8, 9): (80.36, 99.96, 77.88, 0.00, 78.76, 0.00),
        ("cifar10", 0.1, 4, 5): (86.08, 80.27, 79.77, 28.22, 78.10, 29.01), ("cifar10", 0.1, 3, 6): (87.19, 91.70, 80.15, 3.76, 76.67, 2.01),
    },
}


def tag_of(method, ds, zeta, r, f):
    return f"{method}_{ds}_z{str(zeta).replace('.', '')}_{r}-{f}"


def jobs(methods, datasets, zetas, ablation: bool):
    out = []
    for m in methods:
        for ds in datasets:
            for z in zetas:
                for r, f in PAIRS[z]:
                    # 논문 절제(그림 11·12)는 SVHN ζ=0.1 에서만
                    variants = ["full", "no_lf", "no_dg"] if (ablation and ds == "svhn" and z == 0.1) else ["full"]
                    out.append(dict(method=m, ds=ds, zeta=z, r=r, f=f, variants=variants, tag=tag_of(m, ds, z, r, f)))
    return out


def command(j, stage="pair", extra=()):
    vs = "[" + ",".join(f'"{v}"' for v in j["variants"]) + "]"
    return [PY, "-X", "utf8", str(ROOT / "run.py"), "--config", str(ROOT / METHODS[j["method"]]), "--stage", stage,
            "--tag", j["tag"], "--set", f'data.name="{j["ds"]}"', "--set", f"data.zeta={j['zeta']}",
            "--set", f"pair.retained={j['r']}", "--set", f"pair.forget={j['f']}",
            "--set", f"data.backdoor_client={j['f']}", "--set", f"run.variants={vs}", *extra]


def done(j) -> bool:
    p = ROOT / "results" / j.get("res_tag", j["tag"]) / "results.csv"
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as fh:
        models = {r["model"] for r in csv.DictReader(fh)}
    return all(f"unlearn_{v}" in models for v in j["variants"])


def run_job(j, extra=(), stage="pair", check=done):
    if check(j):
        print(f"[skip] {j['tag']} (이미 있음)", flush=True)
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    t0 = time.time()
    print(f"[start] {j['tag']}", flush=True)
    with open(ROOT / "logs" / f"{j['tag']}_{stage}.out", "a", encoding="utf-8") as fh:
        rc = subprocess.call(command(j, stage, extra), stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"[{'done' if rc == 0 else f'FAIL rc={rc}'}] {j['tag']}  {(time.time() - t0) / 60:.1f} min", flush=True)


def prepare(datasets):
    """데이터를 미리 한 번씩 받아 둔다 (작업을 동시에 돌릴 때 같은 파일을 동시에 받다 깨지는 것 방지)."""
    from zfu.data import DataCfg, load
    for ds in datasets:
        print(f"[data] {ds}", flush=True)
        load(DataCfg(name=ds, root=str(ROOT / "data")))


# ---------------------------------------------------------------- 시간 추정
def bench(rounds: int, unlearn_rounds: int, idents=("prompt", "labeldist")) -> dict:
    """무작위 데이터로 로컬 학습 한 스텝, 지우기 한 번 반복의 시간을 재서 작업 하나 시간을 계산한다."""
    import torch
    from zfu.fl import FLCfg, local_update
    from zfu.model import Net
    from zfu.unlearn import UnlearnCfg, Unlearner
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for ident in idents:
        for ds, (n, ch, img) in SIZES.items():
            m = Net(ident, ch, img, 10).to(dev)
            x = torch.randn(32 * 40, ch, img, img, device=dev)
            y = torch.randint(0, 10, (32 * 40,), device=dev)
            c = torch.randn(512, device=dev) if ident == "prompt" else torch.full((10,), 0.1, device=dev)
            cfg = FLCfg(local_epochs=1)
            g = torch.Generator().manual_seed(0)
            local_update(m, c, x[:320], y[:320], cfg, g)
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            local_update(m, c, x, y, cfg, g)
            if dev == "cuda":
                torch.cuda.synchronize()
            n_steps = 40 * (2 if ident == "prompt" else 1)        # prompt 는 에폭 하나 더 (G_p 갱신)
            t_step = (time.time() - t) / n_steps
            ucfg = UnlearnCfg(log_every=10 ** 9)
            un = Unlearner(m, m, c, c, lambda: Net(ident, ch, img, 10).to(dev), ch, img, ucfg, 0, log=lambda *_: None)
            un.run(3)
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            un.run(10)
            if dev == "cuda":
                torch.cuda.synchronize()
            t_un = (time.time() - t) / 10
            steps_round = n / 32 * (3 + (1 if ident == "prompt" else 0))
            fl_run = rounds * steps_round * t_step
            out[(ident, ds)] = dict(fl_run=fl_run, unlearn=unlearn_rounds * t_un)
    return out


def estimate(js, workers: int, rounds: int, unlearn_rounds: int, continue_jobs=()):
    print("측정 중 (1~2분)...", flush=True)
    b = bench(rounds, unlearn_rounds)
    ident = {"mimir": "prompt", "zerofu": "labeldist"}
    total = 0.0
    print(f"\n{'작업':34s} {'연합학습 2회':>12s} {'지우기':>8s} {'합계':>8s}")
    for j in js:
        e = b[(ident[j["method"]], j["ds"])]
        sec = 2 * e["fl_run"] + len(j["variants"]) * e["unlearn"]
        total += sec
        print(f"{j['tag']:34s} {2 * e['fl_run'] / 60:10.1f}분 {len(j['variants']) * e['unlearn'] / 60:6.1f}분 {sec / 60:6.1f}분")
    for j in continue_jobs:
        e = b[(ident[j["method"]], j["ds"])]
        n_ret = 9
        sec = (1 + 2 * n_ret + n_ret) * e["unlearn"] + j["protocols"] * j["cont_rounds"] / rounds * e["fl_run"]
        total += sec
        print(f"{j['tag'] + ' [주제A]':34s} {'':>12s} {'':>8s} {sec / 60:6.1f}분")
    print(f"\n합계 {total / 3600:.1f} 시간 (작업 1개씩). --workers {workers} 이면 대략 {total / 3600 / max(1, workers) * 1.15:.1f} 시간"
          " (CPU 코어가 충분할 때; 이 코드는 CPU 오버헤드가 커서 GPU 가 좋아도 크게 줄지 않는다)")


# ---------------------------------------------------------------- 요약
def summary(js):
    lines = ["# 재현 요약", "",
             "논문 수치와 우리 재현을 나란히 둔다. 단위 %. Dirichlet 분할 난수가 논문과 달라 C_r·C_f 가 가진 클래스가 다르므로 같은 칸끼리도 대략 비교만 된다.",
             "Origin Df 는 C_f 자기 이름표로, 나머지 Df 는 C_r 이름표로 잰 값. **Origin(C_r 이름표) Df** 는 지우기 없이 이름표만 바꾼 원래 모델 (우리가 더한 기준선).", "",
             "| 방법 | 데이터 | ζ | C_r | C_f | | Origin Dr | Origin Df | Origin(C_r 이름표) Df | Retrain Dr | Retrain Df | 지우기 Dr | 지우기 Df | ASR 원래→지우기 (재학습) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows_csv = []
    for j in js:
        p = ROOT / "results" / j.get("res_tag", j["tag"]) / "results.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            rs = {r["model"]: r for r in csv.DictReader(fh)}
        if "original" not in rs or "unlearn_full" not in rs:
            continue
        o, rt, u = rs["original"], rs["retrain"], rs["unlearn_full"]
        pct = lambda v: f"{float(v) * 100:.1f}" if v not in (None, "", "nan") else "-"
        ours = [pct(o["dr_acc"]), pct(o.get("df_acc_own")), pct(o["df_acc"]), pct(rt["dr_acc"]), pct(rt["df_acc"]), pct(u["dr_acc"]), pct(u["df_acc"])]
        asr = f"{pct(o.get('asr_own'))}→{pct(u['asr'])} ({pct(rt['asr'])})"
        pp = PAPER[j["method"]].get((j["ds"], j["zeta"], j["r"], j["f"]))
        head = f"| {j['method']} | {j['ds']} | {j['zeta']} | {j['r']} | {j['f']} |"
        lines.append(f"{head} 우리 | {ours[0]} | {ours[1]} | {ours[2]} | {ours[3]} | {ours[4]} | {ours[5]} | {ours[6]} | {asr} |")
        if pp:
            lines.append(f"| | | | | | 논문 | {pp[0]} | {pp[1]} | | {pp[2]} | {pp[3]} | {pp[4]} | {pp[5]} | |")
        for name, r in rs.items():
            rows_csv.append({"method": j["method"], "dataset": j["ds"], "zeta": j["zeta"], **r})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rows_csv:
        cols = []
        for r in rows_csv:
            cols += [k for k in r if k not in cols]
        with open(ROOT / "results" / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows_csv)
    print(f"요약: {ROOT / 'results' / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(METHODS), nargs="*")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--zetas", nargs="*", type=float, default=[0.01, 0.1])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--rounds", type=int, default=20, help="연합학습 라운드 (설정 파일 값 덮어씀)")
    ap.add_argument("--smoke", action="store_true", help="축소 설정으로 배관만 확인")
    a = ap.parse_args()
    js = jobs(a.only or list(METHODS), a.datasets, a.zetas, not a.no_ablation)
    if a.smoke:
        for j in js:
            j["res_tag"] = j["tag"] + "_smoke"
    if a.estimate:
        estimate(js, a.workers, a.rounds, 2000)
        return
    if not a.summary:
        extra = ("--smoke",) if a.smoke else ("--set", f"fl.rounds={a.rounds}")
        prepare(sorted({j["ds"] for j in js}))
        t0 = time.time()
        print(f"작업 {len(js)}개, 동시 {a.workers}개. 로그: logs/<tag>_pair.out", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda j: run_job(j, extra), js))
        print(f"전체 {(time.time() - t0) / 3600:.2f} 시간", flush=True)
    summary(js)


if __name__ == "__main__":
    main()
