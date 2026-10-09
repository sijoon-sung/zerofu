"""continuation.py - 주제 A: 데이터 없이 지운 뒤에도 연합학습은 계속된다.

두 논문(Mimir, ZeroFU)의 지우기는 남는 클라이언트 C_r 한 명과 지울 클라이언트 C_f 한 쌍 사이에서 일어나고,
결과 학생은 "C_r 에서 계속 쓰인다" (Mimir §IV-C). 그 뒤 연합이 어떻게 이어지는지는 두 논문 모두 다루지 않는다.
여기서는 지운 직후 서버가 어떤 모델로 다시 시작하느냐(프로토콜)를 바꿔 가며, 남은 클라이언트들로
연합학습을 C 라운드 더 돌리고 라운드마다 다음을 잰다.

  df_max     C_f 학습 데이터 정확도, 남은 클라이언트 이름표 중 최댓값 (지운 게 되살아나는가)
  df_r       같은 것을 C_r 이름표로
  asr_max    백도어 성공률, 남은 이름표 중 최댓값
  ret_mean / ret_min   남은 클라이언트 각자 자기 이름표·자기 데이터 정확도의 평균 / 최저 (다른 사람이 망가졌나)
  cr_acc     C_r 자기 정확도
  test_mean  공용 시험셋 정확도 (남은 이름표 평균)

프로토콜 (서버가 다시 시작하는 모델)
  none              지우기 없이 원래 모델에서, C_f 만 빼고 계속 (그냥 떠나기 = 미세조정 기준선)
  pair_adopt        논문 그대로 한 쌍 지우기 → 그 학생을 서버가 새 전역 모델로 채택
  all_pairs_avg     남은 클라이언트 전원이 각자 한 쌍 지우기 (논문처럼 각자 무작위 초기화) → 서버가 평균
  all_pairs_shared  같지만 학생 초기값을 모두 같게 (같은 씨앗) → 평균
  fed_distill       같은 초기 학생에서 출발해, 라운드마다 각자 자기 교사로 조금씩 지우기 증류 → 서버 평균 (연합 증류)
  retrain           C_f 없이 처음부터 학습한 모델에서 계속 (기준)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .fl import FLCfg, accuracy, fedavg, run as fl_run
from .model import Net
from .unlearn import Unlearner, UnlearnCfg, make_student


@dataclass
class ContinueCfg:
    rounds: int = 20                 # 지운 뒤 더 돌리는 라운드 C
    fd_rounds: int = 20              # fed_distill 통신 라운드
    fd_local: int = 100              # fed_distill 라운드당 클라이언트별 지우기 반복 수


@torch.no_grad()
def track(model: Net, fed, conds: dict, r: int, f: int, retained: list[int]) -> dict:
    accs = {i: accuracy(model, conds[i], fed.xs[i], fed.ys[i]) for i in retained}
    out = {
        "df_max": max(accuracy(model, conds[i], fed.xs[f], fed.ys[f]) for i in retained),
        "df_r": accuracy(model, conds[r], fed.xs[f], fed.ys[f]),
        "ret_mean": sum(accs.values()) / len(accs),
        "ret_min": min(accs.values()),
        "cr_acc": accs[r],
        "test_mean": sum(accuracy(model, conds[i], fed.x_test, fed.y_test) for i in retained) / len(retained),
    }
    if fed.bd_test is not None:
        out["asr_max"] = max(accuracy(model, conds[i], fed.bd_test[0], fed.bd_test[1]) for i in retained)
    return out


def start_model(protocol: str, exp, orig: Net, oconds: dict, r: int, f: int, retained: list[int],
                ucfg: UnlearnCfg, ccfg: ContinueCfg) -> Net:
    """지운 직후 서버가 갖는 전역 모델. 비용(초)도 exp.log 로 남긴다."""
    build, in_ch, img, seed, log = exp.build, exp.in_ch, exp.img, exp.seed, exp.log
    if protocol == "none":
        return copy.deepcopy(orig)
    if protocol == "pair_adopt":
        return Unlearner(orig, orig, oconds[r], oconds[f], build, in_ch, img, ucfg, seed, log=log).run(ucfg.rounds)
    if protocol in ("all_pairs_avg", "all_pairs_shared"):
        studs = []
        for k, i in enumerate(retained):
            s_seed = seed if protocol == "all_pairs_shared" else seed + 1000 + k
            log(f"    [{protocol}] client {i} ({k + 1}/{len(retained)})")
            un = Unlearner(orig, orig, oconds[i], oconds[f], build, in_ch, img, ucfg, s_seed, log=log)
            studs.append(un.run(ucfg.rounds).state_dict())
        m = build()
        m.load_state_dict(fedavg(studs, [len(exp.fed.ys[i]) for i in retained]))
        return m
    if protocol == "fed_distill":
        glob = make_student(orig, build, seed)
        total = ccfg.fd_rounds * ccfg.fd_local
        # 클라이언트마다 생성기·옵티마이저 상태를 유지하고, 학생 몸통만 라운드마다 서버 평균으로 바꿔 끼운다
        locs = {i: Unlearner(orig, orig, oconds[i], oconds[f], build, in_ch, img, ucfg, seed + 1000 + k,
                             student=copy.deepcopy(glob), total=total, log=lambda *_: None)
                for k, i in enumerate(retained)}
        for t in range(ccfg.fd_rounds):
            states = []
            for i in retained:
                locs[i].student.load_state_dict(glob.state_dict())
                states.append(copy.deepcopy(locs[i].run(ccfg.fd_local).state_dict()))
            glob.load_state_dict(fedavg(states, [len(exp.fed.ys[i]) for i in retained]))
            if (t + 1) % max(1, ccfg.fd_rounds // 5) == 0:
                tr = track(glob, exp.fed, oconds, r, f, retained)
                log(f"    [fed_distill] round {t + 1}/{ccfg.fd_rounds}  " + "  ".join(f"{k}={v:.3f}" for k, v in tr.items()))
        return glob
    raise ValueError(protocol)


def continue_after(exp, protocols: list[str], ccfg: ContinueCfg, flcfg: FLCfg) -> list[dict]:
    pair = exp.cfg.get("pair", {})
    r, f = pair.get("retained", 0), pair.get("forget", 1)
    orig, oconds = exp.pretrain()
    retained = [i for i in exp.all_clients() if i != f]
    rows = []
    for proto in protocols:
        path = exp.ckdir / f"cont_start_{r}-{f}_{proto}.pt"
        if proto == "retrain":
            m, conds = exp.retrain([f])
            m = copy.deepcopy(m)
        else:
            conds = oconds
            m = exp.build()
            if path.exists():
                m.load_state_dict(torch.load(path, map_location=exp.dev))
            else:
                exp.log(f"[continue] start model: {proto}")
                import time
                t0 = time.time()
                m = start_model(proto, exp, orig, oconds, r, f, retained, exp.ucfg, ccfg)
                exp.log(f"  {proto} start model took {time.time() - t0:.1f}s")
                torch.save(m.state_dict(), path)

        def on_round(t, model, cds, proto=proto):
            row = {"protocol": proto, "t": t, **track(model, exp.fed, cds, r, f, retained)}
            rows.append(row)
            if t % max(1, ccfg.rounds // 5) == 0:
                exp.log(f"  [{proto}] t={t}  " + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)))

        exp.log(f"[continue] {proto}: FL {ccfg.rounds} more rounds with {retained}")
        fl_run(m, exp.fed, retained, flcfg, exp.seed + 99, None, exp.log, conds=conds, rounds=ccfg.rounds, on_round=on_round)
    for row in rows:
        row.update({"retained": r, "forget": f})
    return rows
