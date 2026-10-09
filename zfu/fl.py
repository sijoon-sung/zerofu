"""fl.py - 개인화 연합학습 (Mimir 알고리즘 1, ZeroFU 알고리즘 1).

공유(집계) 부품: 백본 θ, 헤드 φ, 이름표 부품(G_p·P_base 또는 eGen·CM).
개인 부품: prompt 일 때만 서술자 d_i (집계 안 함). labeldist 는 LD_i 를 데이터에서 셈.

로컬 학습
  prompt    : (1) T_l 에폭 동안 θ, φ, d_i 를 CE 로 갱신 (G_p, P_base 고정) — Mimir 식 (10)(11)
              (2) 한 에폭 더, G_p, P_base 만 L_GP = CE + λ1‖G_p‖² + λ2‖P_base‖² 로 갱신 — 식 (12)(13)
  labeldist : T_l 에폭 동안 전부를 L = CE + L_EM + λ1‖eGen‖² + λ2‖CM‖² 로 갱신 — ZeroFU 식 (12)~(14)
집계: 표본 수 가중 FedAvg. Mimir 식 (1) 의 λ_g 근접항은 알고리즘 1 에 안 나와서 뺐다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import Net


@dataclass
class FLCfg:
    rounds: int = 20              # 논문에 명시 없음 (Mimir 그림 1 예시가 20 라운드) → 가정
    local_epochs: int = 3         # T_l = 3
    lr: float = 0.005
    momentum: float = 0.0         # 식 (11) 은 순수 경사하강
    batch_size: int = 32
    lam1: float = 0.1
    lam2: float = 0.1
    participation: float = 1.0    # α = 1
    eval_every: int = 5


def batches(n: int, bs: int, gen: torch.Generator, device):
    perm = torch.randperm(n, generator=gen).to(device)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


def sq_norm(params) -> torch.Tensor:
    return sum((p ** 2).sum() for p in params)


def em_loss(fg: torch.Tensor, emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ZeroFU 식 (13): f_g 와 자기 클래스 임베딩의 코사인 유사도 소프트맥스."""
    cos = F.normalize(fg, dim=1) @ F.normalize(emb, dim=1).T
    return F.cross_entropy(cos, y)


def local_update(model: Net, c: torch.Tensor, x, y, cfg: FLCfg, gen: torch.Generator):
    """model 을 제자리에서 갱신. prompt 면 새 서술자를 돌려준다."""
    model.train()
    dev = x.device
    if model.ident == "prompt":
        d = c.clone().requires_grad_(True)
        opt = torch.optim.SGD(model.body_params() + [d], lr=cfg.lr, momentum=cfg.momentum)
        for _ in range(cfg.local_epochs):
            for idx in batches(len(y), cfg.batch_size, gen, dev):
                out, _ = model(x[idx], d)
                loss = F.cross_entropy(out, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        d = d.detach()
        gp = model.gp
        opt2 = torch.optim.SGD(gp.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        others = [p for n, p in gp.named_parameters() if n != "p_base"]
        for idx in batches(len(y), cfg.batch_size, gen, dev):
            out, _ = model(x[idx], d)
            loss = F.cross_entropy(out, y[idx]) + cfg.lam1 * sq_norm(others) + cfg.lam2 * (gp.p_base ** 2).sum()
            opt2.zero_grad()
            loss.backward()
            opt2.step()
        return d
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    for _ in range(cfg.local_epochs):
        for idx in batches(len(y), cfg.batch_size, gen, dev):
            out, aux = model(x[idx], c)
            loss = (F.cross_entropy(out, y[idx]) + em_loss(aux["f_g"], model.cond.embeddings(), y[idx])
                    + cfg.lam1 * sq_norm(model.cond.egen.parameters())
                    + cfg.lam2 * (sq_norm(model.cond.cm_w.parameters()) + sq_norm(model.cond.cm_b.parameters())))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return c


def fedavg(states: list[dict], weights: list[float]) -> dict:
    tot = sum(weights)
    out = {}
    for k in states[0]:
        if states[0][k].dtype.is_floating_point:
            out[k] = sum(s[k] * (w / tot) for s, w in zip(states, weights))
        else:
            out[k] = states[0][k].clone()
    return out


@torch.no_grad()
def accuracy(model: Net, c: torch.Tensor, x, y, bs: int = 1024) -> float:
    model.eval()
    if len(y) == 0:
        return float("nan")
    hit = 0
    for i in range(0, len(y), bs):
        out, _ = model(x[i:i + bs], c)
        hit += (out.argmax(1) == y[i:i + bs]).sum().item()
    return hit / len(y)


def init_conds(model: Net, fed, clients, gen: torch.Generator, device) -> dict[int, torch.Tensor]:
    if model.ident == "prompt":
        dim = model.theta.dim
        return {i: torch.randn(dim, generator=gen).to(device) for i in clients}   # 초기값은 논문에 없음 (가정)
    return {i: fed.ld[i].to(device) for i in clients}


def run(model: Net, fed, clients: list[int], cfg: FLCfg, seed: int, ckpt: Path | None = None, log=print,
        conds: dict | None = None, rounds: int | None = None, on_round=None):
    """clients 로만 연합학습. ckpt 가 있으면 라운드마다 저장하고 이어 간다. (model, conds) 반환.
    conds 를 주면 그 이름표에서 이어 가고 (지운 뒤 계속 학습), on_round(r, model, conds) 는 라운드마다 부른다."""
    dev = next(model.parameters()).device
    for p in model.parameters():          # 지우기 때 고정했던 부품도 다시 학습 대상으로
        p.requires_grad_(True)
    gen = torch.Generator().manual_seed(seed)
    init = init_conds(model, fed, clients, gen, dev)
    conds = {i: (conds[i].clone() if conds is not None and i in conds else init[i]) for i in clients}
    total = rounds if rounds is not None else cfg.rounds
    start = 0
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"])
        conds = {int(k): v.to(dev) for k, v in st["conds"].items()}
        start = st["round"]
        gen.set_state(st["gen"])
        log(f"  resume from round {start}")
    sizes = {i: len(fed.ys[i]) for i in clients}
    if on_round is not None and start == 0:
        on_round(0, model, conds)
    for r in range(start, total):
        n_pick = max(1, round(cfg.participation * len(clients)))
        pick = [clients[j] for j in torch.randperm(len(clients), generator=gen)[:n_pick].tolist()]
        g_state = copy.deepcopy(model.state_dict())
        states, ws = [], []
        for i in pick:
            model.load_state_dict(g_state)
            conds[i] = local_update(model, conds[i], fed.xs[i], fed.ys[i], cfg, gen)
            states.append(copy.deepcopy(model.state_dict()))
            ws.append(sizes[i])
        model.load_state_dict(fedavg(states, ws))
        if on_round is not None:
            on_round(r + 1, model, conds)
        if (r + 1) % cfg.eval_every == 0 or r + 1 == total:
            accs = [accuracy(model, conds[i], fed.xs[i], fed.ys[i]) for i in clients]
            log(f"  round {r + 1}/{total}  mean local acc {sum(accs) / len(accs):.4f}")
        if ckpt is not None:
            torch.save({"model": model.state_dict(), "conds": conds, "round": r + 1, "gen": gen.get_state()}, ckpt)
    return model, conds
