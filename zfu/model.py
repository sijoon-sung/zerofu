"""model.py - 백본 CNN + 두 가지 클라이언트 조건(이름표) + 헤드.

백본 θ (Mimir §VI-A-2, ZeroFU §4.1 같은 CNN):
  conv(32, 5×5) - ReLU - maxpool2 - conv(64, 5×5) - ReLU - maxpool2 - flatten - FC(512) - ReLU  → f_θ
헤드 φ: 마지막 FC (클래스 수로).

이름표 두 종류 (ident):
  "prompt"    Mimir 식 (7)(8). 클라이언트마다 학습되는 서술자 d_i (집계 안 함, 모델 밖에 보관).
              P_i = P_base + [Softmax(Q Kᵀ/√l_k) V W_proj]ᵀ,  Q = d_i W_Q, K = P_baseᵀ W_K, V = P_baseᵀ W_V
              (특징 차원 하나하나를 토큰으로 보는 꼴: d_i, P_base ∈ R^{l_θ}, W_Q·W_K ∈ R^{1×l_k}, W_V ∈ R^{1×l_v})
              f_P = ‖f_θ‖ / ‖P_i ∘ f_θ‖ · (P_i ∘ f_θ),  ŷ = φ(f_P)
  "labeldist" ZeroFU 식 (4)~(11) (GPFL 의 조건 계산). 조건 = 라벨 분포 LD_i (데이터에서 셈, 학습 안 함).
              cEm = eGen(클래스), gEm = 평균 cEm, pEm_i = Σ_u LD_i^u cEm_u
              W, b = CM(gEm) → f_g = ReLU(b + (W+1)⊙f_θ);  W_i, b_i = CM(pEm_i) → f_p 같은 꼴
              ŷ = φ([f_p; f_g])

forward 는 (logits, aux) 를 돌려준다. aux["feat"] 은 지우기 손실 LF 에 쓰는 개인 특징
(prompt: f_P, labeldist: f_p), aux["acts"] 는 어텐션 손실용 합성곱 활성값.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Backbone(nn.Module):
    def __init__(self, in_ch: int, img: int, dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        s = ((img - 4) // 2 - 4) // 2
        self.fc = nn.Linear(64 * s * s, dim)
        self.dim = dim

    def forward(self, x):
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(F.max_pool2d(a1, 2)))
        f = F.relu(self.fc(F.max_pool2d(a2, 2).flatten(1)))
        return f, [a1, a2]


class PromptGen(nn.Module):
    """Mimir 프롬프트 생성기 G_p 와 P_base (공유, 집계됨)."""

    def __init__(self, dim: int, lk: int = 16, lv: int = 16):
        super().__init__()
        # 초기값은 논문에 없다. 식 (7) 을 글자 그대로 옮기면 특징 차원 하나가 토큰 하나라 점수가 d_ij·P_base,l·(w_q·w_k) 꼴이 되어,
        # P_base 를 1 근처로 두거나 투영을 작게 두면 클라이언트마다 프롬프트가 똑같아진다 (분산 1e-6, 확인함).
        # 그래서 P_base ~ N(0,1), 투영 ~ N(0,1) 로 둔다 (같은 조건에서 클라이언트 간 분산 ≈ 10).
        self.p_base = nn.Parameter(torch.randn(dim))
        self.wq = nn.Parameter(torch.randn(1, lk))
        self.wk = nn.Parameter(torch.randn(1, lk))
        self.wv = nn.Parameter(torch.randn(1, lv))
        self.wproj = nn.Parameter(torch.randn(lv, 1))
        self.lk = lk

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        q = d.view(-1, 1) @ self.wq                    # l_θ × l_k
        k = self.p_base.view(-1, 1) @ self.wk          # l_θ × l_k
        v = self.p_base.view(-1, 1) @ self.wv          # l_θ × l_v
        att = torch.softmax(q @ k.T / math.sqrt(self.lk), dim=-1)
        return self.p_base + (att @ v @ self.wproj).view(-1)


class LabelCond(nn.Module):
    """ZeroFU 의 eGen (클래스 임베딩) 과 조건 모듈 CM (공유, 집계됨)."""

    def __init__(self, n_classes: int, dim: int):
        super().__init__()
        self.egen = nn.Embedding(n_classes, dim)
        self.cm_w = nn.Linear(dim, dim)
        self.cm_b = nn.Linear(dim, dim)

    def embeddings(self) -> torch.Tensor:
        return self.egen.weight                         # U × dim

    def transform(self, f: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return F.relu(self.cm_b(emb) + (self.cm_w(emb) + 1) * f)


class Net(nn.Module):
    def __init__(self, ident: str, in_ch: int, img: int, n_classes: int, dim: int = 512,
                 lk: int = 16, lv: int = 16):
        super().__init__()
        self.ident = ident
        self.theta = Backbone(in_ch, img, dim)
        if ident == "prompt":
            self.gp = PromptGen(dim, lk, lv)
            self.head = nn.Linear(dim, n_classes)
        elif ident == "labeldist":
            self.cond = LabelCond(n_classes, dim)
            self.head = nn.Linear(2 * dim, n_classes)
        else:
            raise ValueError(ident)

    # 이름표에 해당하는 공유 부품 (지우기 때 학생에게 복사·고정)
    def ident_params(self) -> list[nn.Parameter]:
        mod = self.gp if self.ident == "prompt" else self.cond
        return list(mod.parameters())

    def body_params(self) -> list[nn.Parameter]:
        return list(self.theta.parameters()) + list(self.head.parameters())

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """c: prompt 이면 서술자 d_i (l_θ), labeldist 이면 라벨 분포 LD_i (U)."""
        f, acts = self.theta(x)
        if self.ident == "prompt":
            p = self.gp(c)
            pf = p * f
            fp = pf * (f.norm(dim=1, keepdim=True) / pf.norm(dim=1, keepdim=True).clamp_min(1e-8))
            return self.head(fp), {"feat": fp, "f_theta": f, "acts": acts}
        emb = self.cond.embeddings()
        gem = emb.mean(0)
        pem = c @ emb
        fg = self.cond.transform(f, gem)
        fpers = self.cond.transform(f, pem)
        return self.head(torch.cat([fpers, fg], 1)), {"feat": fpers, "f_g": fg, "f_theta": f, "acts": acts}


class Generator(nn.Module):
    """잡음 → 가짜 이미지 (Micaelli & Storkey 2019 ZSKT 생성기 꼴; 두 논문 서술과 같은 층 구성)."""

    def __init__(self, z_dim: int, out_ch: int, img: int, ngf: int = 128):
        super().__init__()
        self.init = img // 4
        self.fc = nn.Linear(z_dim, ngf * self.init * self.init)
        self.ngf = ngf
        self.net = nn.Sequential(
            nn.BatchNorm2d(ngf),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, padding=1), nn.BatchNorm2d(ngf), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf // 2, 3, padding=1), nn.BatchNorm2d(ngf // 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf // 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, affine=False),
        )
        self.img = img

    def forward(self, z):
        h = self.fc(z).view(-1, self.ngf, self.init, self.init)
        x = self.net(h)
        if x.shape[-1] != self.img:          # MNIST 28 = 7×4 이라 맞음, 혹시 안 맞으면 보간
            x = F.interpolate(x, size=self.img, mode="bilinear", align_corners=False)
        return x
