"""experiment.py - 단계별 실행과 결과 기록.

단계
  pretrain   전원으로 개인화 연합학습 → original.pt
  retrain    지울 클라이언트를 뺀 나머지로 처음부터 → retrain_<빠진 목록>.pt  (기준 모델)
  unlearn    (C_r, C_f) 한 쌍에 대해 지우기 변형들(full / no_lf / no_dg) → unlearned_<변형>.pt, 결과 CSV
  sequential 연속 삭제 (후속 주제용 틀): C_f 를 차례로 지우며 매번 재학습과 비교
  continue   주제 A: 지운 직후 서버가 어떤 모델로 시작하느냐(프로토콜)별로 연합학습을 더 돌리며 되살아남 추적
산출물이 이미 있으면 그 단계는 건너뛴다 (다시 하려면 파일을 지운다). 연합학습은 라운드마다 저장하고 이어 간다.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

from . import data as D
from .fl import FLCfg, accuracy, run as fl_run
from .metrics import evaluate
from .model import Net
from .unlearn import UnlearnCfg, unlearn
from .continuation import ContinueCfg, continue_after


# 지우기 변형: full = 설정 그대로, no_lf = 그림 11(b), no_dg = 그림 12(a),
# gen_lf / gen_lflk = 생성기 목표를 ZeroFU 식 (18) / Mimir 식 (19) 로 강제 (두 논문 차이 확인용)
VARIANTS = {"no_lf": {"use_lf": False}, "no_dg": {"use_dg": False}, "gen_lf": {"gen_obj": "lf"}, "gen_lflk": {"gen_obj": "lf_lk"}}


def _fill(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


class Experiment:
    def __init__(self, cfg: dict, root: Path):
        self.cfg = cfg
        self.tag = cfg["tag"]
        self.seed = cfg.get("seed", 0)
        self.dev = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dcfg = _fill(D.DataCfg, cfg["data"])
        if not Path(self.dcfg.root).is_absolute():          # 상대 경로는 저장소 기준
            self.dcfg.root = str(root / self.dcfg.root)
        self.flcfg = _fill(FLCfg, cfg.get("fl", {}))
        self.ucfg = _fill(UnlearnCfg, cfg.get("unlearn", {}))
        self.mcfg = cfg.get("model", {})
        self.ckdir = root / "checkpoints" / self.tag
        self.resdir = root / "results" / self.tag
        self.ckdir.mkdir(parents=True, exist_ok=True)
        self.resdir.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        self.logf = open(root / "logs" / f"{self.tag}.log", "a", encoding="utf-8")
        (self.resdir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"=== {self.tag}  device {self.dev}")
        fed = D.build(self.dcfg)
        fed.xs = [x.to(self.dev) for x in fed.xs]
        fed.ys = [y.to(self.dev) for y in fed.ys]
        fed.x_test, fed.y_test = fed.x_test.to(self.dev), fed.y_test.to(self.dev)
        if fed.bd_test is not None:
            fed.bd_test = (fed.bd_test[0].to(self.dev), fed.bd_test[1].to(self.dev))
        self.fed = fed
        self.in_ch, self.img = fed.xs[0].shape[1], fed.xs[0].shape[-1]
        self.log(f"  client sizes {fed.sizes}")
        self.log("  label hist " + json.dumps([torch.bincount(y.cpu(), minlength=fed.n_classes).tolist() for y in fed.ys]))

    # ------------------------------------------------------------------ 공통
    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.logf:
            self.logf.write(line + "\n")
            self.logf.flush()

    def build(self) -> Net:
        return Net(self.mcfg.get("ident", "prompt"), self.in_ch, self.img, self.fed.n_classes,
                   self.mcfg.get("dim", 512), self.mcfg.get("lk", 16), self.mcfg.get("lv", 16)).to(self.dev)

    def _train(self, clients: list[int], name: str):
        path = self.ckdir / f"{name}.pt"
        model = self.build()
        if path.exists():
            st = torch.load(path, map_location=self.dev)
            model.load_state_dict(st["model"])
            return model, {int(k): v for k, v in st["conds"].items()}
        self.log(f"[{name}] FL with clients {clients}")
        torch.manual_seed(self.seed)
        model = self.build()
        model, conds = fl_run(model, self.fed, clients, self.flcfg, self.seed, self.ckdir / f"prog_{name}.pt", self.log)
        torch.save({"model": model.state_dict(), "conds": conds}, path)
        (self.ckdir / f"prog_{name}.pt").unlink(missing_ok=True)
        return model, conds

    def _append(self, rows: list[dict], fname: str = "results.csv"):
        path = self.resdir / fname
        old = []
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                old = list(csv.DictReader(fh))
        def key(r):
            return tuple(str(r.get(k, "")) for k in ("model", "step", "protocol", "t", "retained", "forget"))
        keys = {key(r) for r in rows}
        old = [r for r in old if key(r) not in keys]
        allrows = old + [{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
        cols = []
        for r in allrows:
            cols += [k for k in r if k not in cols]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(allrows)

    def all_clients(self) -> list[int]:
        return list(range(self.dcfg.n_clients))

    # ------------------------------------------------------------------ 단계
    def pretrain(self):
        return self._train(self.all_clients(), "original")

    def retrain(self, forgotten: list[int]):
        keep = [i for i in self.all_clients() if i not in forgotten]
        return self._train(keep, "retrain_" + "-".join(map(str, sorted(forgotten))))

    def unlearn_pair(self):
        pair = self.cfg.get("pair", {})
        r, f = pair.get("retained", 0), pair.get("forget", 1)
        variants = self.cfg.get("run", {}).get("variants", ["full"])
        orig, oconds = self.pretrain()
        retr, rconds = self.retrain([f])
        retained = [i for i in self.all_clients() if i != f]
        rows = [
            {"model": "original", **evaluate(orig, self.fed, oconds, r, f, retained, self.seed, retr, own_cond=oconds[f])},
            {"model": "retrain", **evaluate(retr, self.fed, rconds, r, f, retained, self.seed, retr)},
        ]
        for v in variants:
            ucfg = UnlearnCfg(**{**asdict(self.ucfg), **VARIANTS.get(v, {})})
            path = self.ckdir / f"unlearned_{r}-{f}_{v}.pt"
            stu = self.build()
            if path.exists():
                stu.load_state_dict(torch.load(path, map_location=self.dev))
                sec = float("nan")
            else:
                self.log(f"[unlearn] C_r={r} C_f={f} variant={v}")
                t0 = time.time()
                stu = unlearn(orig, orig, oconds[r], oconds[f], self.build, self.in_ch, self.img, ucfg, self.seed, self.log)
                sec = time.time() - t0
                torch.save(stu.state_dict(), path)
            rows.append({"model": f"unlearn_{v}", "seconds": sec,
                         **evaluate(stu, self.fed, oconds, r, f, retained, self.seed, retr)})
        for row in rows:
            row.update({"retained": r, "forget": f})
            self.log("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        self._append(rows)

    def sequential(self):
        """연속 삭제 틀: forget 목록을 차례로 지운다. 매 단계 교사 T_R = C_r 의 현재 모델 (앞 단계 학생).
        tf_source = original 이면 T_F 는 원래 모델 (지울 클라이언트는 학생을 받은 적 없음), current 면 현재 모델."""
        sc = self.cfg.get("sequential", {})
        r = sc.get("retained", 0)
        order = sc.get("forget", [1, 2, 3])
        tf_source = sc.get("tf_source", "original")
        orig, oconds = self.pretrain()
        cur = orig
        rows = []
        for k, f in enumerate(order):
            gone = order[:k + 1]
            retr, rconds = self.retrain(gone)
            retained = [i for i in self.all_clients() if i not in gone]
            path = self.ckdir / f"seq_{r}_{'-'.join(map(str, gone))}_{tf_source}.pt"
            stu = self.build()
            if path.exists():
                stu.load_state_dict(torch.load(path, map_location=self.dev))
            else:
                self.log(f"[sequential] step {k + 1}: forget {f} (gone {gone})")
                tf = orig if tf_source == "original" else cur
                stu = unlearn(cur, tf, oconds[r], oconds[f], self.build, self.in_ch, self.img, self.ucfg, self.seed + k, self.log)
                torch.save(stu.state_dict(), path)
            cur = stu
            row = {"model": f"seq_{tf_source}", "step": k + 1, "forget": f, "retained": r,
                   **evaluate(stu, self.fed, oconds, r, f, retained, self.seed, retr)}
            # 앞서 지운 클라이언트들이 다시 살아나는지
            for j in gone[:-1]:
                row[f"df_acc_prev{j}"] = accuracy(stu, oconds[r], self.fed.xs[j], self.fed.ys[j])
            rrow = {"model": "retrain_seq", "step": k + 1, "forget": f, "retained": r,
                    **evaluate(retr, self.fed, rconds, r, f, retained, self.seed, retr)}
            rows += [row, rrow]
            self.log("  " + json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}))
        self._append(rows, "sequential.csv")

    def continue_stage(self):
        cc = self.cfg.get("continue", {})
        ccfg = _fill(ContinueCfg, cc)
        protocols = cc.get("protocols", ["none", "pair_adopt", "retrain"])
        rows = continue_after(self, protocols, ccfg, self.flcfg)
        self._append(rows, "continue.csv")
