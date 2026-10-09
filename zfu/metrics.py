"""metrics.py - 두 논문이 쓰는 지표.

  dr_acc     남는 클라이언트 C_r 학습 데이터 정확도 (C_r 이름표)
  df_acc     지울 클라이언트 C_f 학습 데이터 정확도 (C_r 이름표; 지운 모델은 C_r 에서 계속 쓰이므로)
  df_acc_max C_f 데이터를 남은 클라이언트 이름표 전부로 재 본 최댓값 (공격자에게 유리한 쪽, 우리가 더함)
  df_acc_own 원래 모델에서만: C_f 자기 이름표로 잰 값
  test_acc   공용 시험셋 정확도 (C_r 이름표)
  asr        백도어 성공률 (트리거 시험셋이 목표 라벨로 분류되는 비율)
  mia_*      멤버십 추론: 논문대로 C_r 데이터(멤버) vs 시험셋(비멤버) 의 개인특징으로 로지스틱 회귀를 학습,
             C_f 데이터(멤버로 쳐야 할 것) + 다른 시험셋 반쪽(비멤버) 에 적용해 정밀도·재현율.
             (논문은 평가 집합 구성을 안 적어서 균형 집합으로 둠)
  wdist      재학습 모델과의 θ·φ 가중치 L2 거리 (Mimir 그림 8 은 층별, 여기선 합)
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .fl import accuracy
from .model import Net


@torch.no_grad()
def features(model: Net, c, x, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(x[i:i + bs], c)[1]["feat"].float().cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy()


def mia(model: Net, c_r, x_r, x_f, x_test, seed: int, cap: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_test))
    half = len(perm) // 2
    te_a, te_b = perm[:half], perm[half:]
    n1 = min(cap, len(x_r), half)
    ir = rng.choice(len(x_r), n1, replace=False)
    ia = rng.choice(te_a, n1, replace=False)
    X = np.concatenate([features(model, c_r, x_r[torch.as_tensor(ir)]), features(model, c_r, x_test[torch.as_tensor(ia)])])
    Y = np.concatenate([np.ones(n1), np.zeros(n1)])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), Y)
    n2 = min(cap, len(x_f), len(te_b))
    jf = rng.choice(len(x_f), n2, replace=False)
    jb = rng.choice(te_b, n2, replace=False)
    Xe = np.concatenate([features(model, c_r, x_f[torch.as_tensor(jf)]), features(model, c_r, x_test[torch.as_tensor(jb)])])
    Ye = np.concatenate([np.ones(n2), np.zeros(n2)])
    pred = clf.predict(sc.transform(Xe))
    tp = ((pred == 1) & (Ye == 1)).sum()
    prec = tp / max((pred == 1).sum(), 1)
    rec = tp / n2
    return {"mia_prec": float(prec), "mia_rec": float(rec), "mia_acc": float((pred == Ye).mean())}


@torch.no_grad()
def asr(model: Net, c, bd) -> float:
    if bd is None:
        return float("nan")
    return accuracy(model, c, bd[0], bd[1])


def wdist(a: Net, b: Net) -> float:
    pa = torch.cat([p.detach().flatten() for p in a.body_params()])
    pb = torch.cat([p.detach().flatten() for p in b.body_params()])
    return float((pa - pb).norm())


def evaluate(model: Net, fed, conds: dict, r: int, f: int, retained: list[int], seed: int,
             ref: Net | None = None, own_cond=None) -> dict:
    c_r = conds[r]
    out = {
        "dr_acc": accuracy(model, c_r, fed.xs[r], fed.ys[r]),
        "df_acc": accuracy(model, c_r, fed.xs[f], fed.ys[f]),
        "df_acc_max": max(accuracy(model, conds[i], fed.xs[f], fed.ys[f]) for i in retained),
        "test_acc": accuracy(model, c_r, fed.x_test, fed.y_test),
        "asr": asr(model, c_r, fed.bd_test),
    }
    if own_cond is not None:
        out["df_acc_own"] = accuracy(model, own_cond, fed.xs[f], fed.ys[f])
        out["asr_own"] = asr(model, own_cond, fed.bd_test)
    out.update(mia(model, c_r, fed.xs[r], fed.xs[f], fed.x_test, seed))
    if ref is not None:
        out["wdist"] = wdist(model, ref)
    return out
