"""data.py - 데이터 읽기와 연합 분할 (Mimir §VI-A, ZeroFU §4.1 설정).

분할 방식 (두 논문 공통):
  pathological : 클라이언트마다 라벨 몇 종만 (McMahan 2017 식 샤드)
  dirichlet    : 클래스마다 Dir(zeta) 비율로 클라이언트에 나눔. zeta = 0.01 / 0.1

백도어 (Mimir §VI-A-4): 지울 클라이언트 Cf 데이터에서 가장 많은 클래스 사진 일부에
오른쪽 아래 3×3 흰 사각형을 찍고 라벨을 (y+1) % U 로 바꾼다.

전부 텐서로 메모리에 올린다 (MNIST·CIFAR-10 크기면 충분).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torchvision import datasets

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fmnist": ((0.2860,), (0.3530,)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


@dataclass
class DataCfg:
    name: str = "mnist"
    root: str = "./data"
    n_clients: int = 10
    partition: str = "dirichlet"     # dirichlet | pathological
    zeta: float = 0.1                # Dirichlet 농도
    shards_per_client: int = 2       # pathological 일 때
    min_size: int = 20
    backdoor_client: int | None = None
    backdoor_frac: float = 0.3       # Cf 의 최다 클래스 중 트리거를 찍는 비율 (논문에 없음, 가정)
    seed: int = 0
    subset: int | None = None        # 스모크용: 학습 데이터 앞부분만


def _to_tensor(x: np.ndarray, name: str) -> torch.Tensor:
    t = torch.as_tensor(x).float() / 255.0
    if t.dim() == 3:                 # N,H,W (흑백)
        t = t.unsqueeze(1)
    elif t.shape[-1] in (1, 3):      # N,H,W,C
        t = t.permute(0, 3, 1, 2)
    m, s = STATS[name]
    return (t - torch.tensor(m).view(1, -1, 1, 1)) / torch.tensor(s).view(1, -1, 1, 1)


def load(cfg: DataCfg):
    """(x_train, y_train, x_test, y_test). 정규화까지 마친 텐서."""
    n = cfg.name
    if n == "mnist":
        tr = datasets.MNIST(cfg.root, train=True, download=True)
        te = datasets.MNIST(cfg.root, train=False, download=True)
        xs = (tr.data.numpy(), te.data.numpy())
        ys = (tr.targets, te.targets)
    elif n == "fmnist":
        tr = datasets.FashionMNIST(cfg.root, train=True, download=True)
        te = datasets.FashionMNIST(cfg.root, train=False, download=True)
        xs = (tr.data.numpy(), te.data.numpy())
        ys = (tr.targets, te.targets)
    elif n == "cifar10":
        tr = datasets.CIFAR10(cfg.root, train=True, download=True)
        te = datasets.CIFAR10(cfg.root, train=False, download=True)
        xs = (tr.data, te.data)
        ys = (torch.tensor(tr.targets), torch.tensor(te.targets))
    elif n == "svhn":
        tr = datasets.SVHN(cfg.root, split="train", download=True)
        te = datasets.SVHN(cfg.root, split="test", download=True)
        xs = (tr.data.transpose(0, 2, 3, 1), te.data.transpose(0, 2, 3, 1))
        ys = (torch.tensor(tr.labels), torch.tensor(te.labels))
    else:
        raise ValueError(n)
    xtr, xte = _to_tensor(xs[0], n), _to_tensor(xs[1], n)
    ytr, yte = torch.as_tensor(ys[0]).long(), torch.as_tensor(ys[1]).long()
    if cfg.subset:
        xtr, ytr = xtr[: cfg.subset], ytr[: cfg.subset]
    return xtr, ytr, xte, yte


def split(y: torch.Tensor, cfg: DataCfg) -> list[np.ndarray]:
    """클라이언트별 학습 인덱스 목록."""
    rng = np.random.default_rng(cfg.seed)
    y = y.numpy()
    U = int(y.max()) + 1
    K = cfg.n_clients
    if cfg.partition == "pathological":
        order = np.argsort(y, kind="stable")
        shards = np.array_split(order, K * cfg.shards_per_client)
        perm = rng.permutation(len(shards))
        return [np.concatenate([shards[j] for j in perm[i::K]]) for i in range(K)]
    # Dirichlet: 최소 크기를 넘을 때까지 다시 뽑는다 (Lin et al. 2020 관행)
    for _ in range(1000):
        parts = [[] for _ in range(K)]
        for c in range(U):
            idx = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet([cfg.zeta] * K)
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        if min(len(p) for p in parts) >= cfg.min_size:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError("Dirichlet 분할이 min_size 를 못 맞춤")


def label_dist(y: torch.Tensor, U: int) -> torch.Tensor:
    """ZeroFU 식 (6): 클라이언트 라벨 비율 LD_i."""
    return torch.bincount(y, minlength=U).float() / max(len(y), 1)


def white(name: str) -> torch.Tensor:
    """정규화 공간에서 흰색(원래 1.0) 값, 채널별."""
    m, s = STATS[name]
    return (1 - torch.tensor(m)) / torch.tensor(s)


def add_trigger(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """오른쪽 아래 3×3 흰 사각형 (N,C,H,W)."""
    x = x.clone()
    x[:, :, -3:, -3:] = w.view(1, -1, 1, 1)
    return x


@dataclass
class Federation:
    xs: list[torch.Tensor]
    ys: list[torch.Tensor]
    x_test: torch.Tensor
    y_test: torch.Tensor
    n_classes: int
    ld: list[torch.Tensor] = field(default_factory=list)   # 라벨 분포 (ZeroFU 조건)
    bd_test: tuple[torch.Tensor, torch.Tensor] | None = None   # 백도어 시험 (트리거 찍힌 x, 목표 라벨)

    @property
    def sizes(self) -> list[int]:
        return [len(y) for y in self.ys]


def build(cfg: DataCfg) -> Federation:
    xtr, ytr, xte, yte = load(cfg)
    U = int(ytr.max()) + 1
    parts = split(ytr, cfg)
    xs = [xtr[torch.as_tensor(p)] for p in parts]
    ys = [ytr[torch.as_tensor(p)] for p in parts]
    bd_test = None
    if cfg.backdoor_client is not None:
        f = cfg.backdoor_client
        src = int(torch.bincount(ys[f], minlength=U).argmax())
        tgt = (src + 1) % U
        g = torch.Generator().manual_seed(cfg.seed)
        cand = torch.where(ys[f] == src)[0]
        pick = cand[torch.randperm(len(cand), generator=g)[: int(len(cand) * cfg.backdoor_frac)]]
        xs[f] = xs[f].clone()
        ys[f] = ys[f].clone()
        w = white(cfg.name)
        xs[f][pick] = add_trigger(xs[f][pick], w)
        ys[f][pick] = tgt
        te_src = xte[yte == src]
        bd_test = (add_trigger(te_src, w), torch.full((len(te_src),), tgt, dtype=torch.long))
    fed = Federation(xs, ys, xte, yte, U, [label_dist(y, U) for y in ys], bd_test)
    return fed
