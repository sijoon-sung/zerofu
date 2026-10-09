"""unlearn.py - 데이터 없는 지우기: GAN + 지식 증류 (Mimir 알고리즘 2, ZeroFU 알고리즘 2).

참여자 (두 논문 공통, 남는 클라이언트 C_r 한 명과 지울 클라이언트 C_f 한 쌍):
  교사 T_R  = 원래 모델 + C_r 이름표
  지울 교사 T_F = 원래 모델 + C_f 이름표
  학생 S    = 무작위 초기화, 이름표 부품은 원래 것을 복사해 고정, C_r 이름표를 씀. 학습 대상은 θ, φ 뿐.
  생성기 DG = 잡음 → 가짜 이미지

반복 (T_u 번):
  x = DG(z)
  T_k 번: S 를 L_S = L_F + β L_K + γ L_att 로 내림
     L_F   = 1 − cos(S 개인특징, T_F 개인특징)        Mimir 식 (15), ZeroFU 식 (15)
     L_K   = τ² KL(σ(T_R/τ) ‖ σ(S/τ))                  식 (16). Mimir 본문은 T 를 T_F 로 적었다가 "T_R 과 S 사이"라고 해서 T_R 로 둠
     L_att = 합성곱 활성값 어텐션 지도 차이               식 (17), Zagoruyko·Micaelli 꼴
  DG 를 올림: Mimir 는 L_DG = L_F + β L_K (식 19), ZeroFU 는 L_F 만 (식 18)

논문에 값이 없는 것 (가정, 설정으로 바꿈): T_u, 배치, 옵티마이저·학습률 (ZSKT 기본값 Adam 2e-3/1e-3, 코사인).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import Generator, Net


@dataclass
class UnlearnCfg:
    rounds: int = 2000          # T_u (가정)
    kd_steps: int = 9           # T_k = 9 (ZeroFU 4.1)
    batch_size: int = 128       # 가정 (ZSKT)
    z_dim: int = 100
    lr_s: float = 2e-3
    lr_g: float = 1e-3
    tau: float = 2.0
    beta: float = 5.0
    gamma: float = 2.0
    gen_obj: str = "auto"       # auto | lf_lk (Mimir) | lf (ZeroFU)
    use_lf: bool = True         # False = Mimir 그림 11 (b) "w/o L_F"
    use_dg: bool = True         # False = Mimir 그림 12 (a) "w/o DG" (생성기 대신 무작위 잡음)
    log_every: int = 200


def att_map(a: torch.Tensor) -> torch.Tensor:
    return F.normalize(a.pow(2).mean(1).flatten(1), dim=1)


def att_loss(acts_t, acts_s) -> torch.Tensor:
    return sum((att_map(t) - att_map(s)).pow(2).mean() for t, s in zip(acts_t, acts_s)) / len(acts_t)


def kd_loss(t_logits, s_logits, tau: float) -> torch.Tensor:
    return F.kl_div(F.log_softmax(s_logits / tau, 1), F.softmax(t_logits / tau, 1), reduction="batchmean") * tau ** 2


def lf_loss(fs: torch.Tensor, ff: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(fs, ff, dim=1)).mean()


def make_student(orig: Net, build, seed: int) -> Net:
    torch.manual_seed(seed)
    s = build().to(next(orig.parameters()).device)
    # 이름표 부품은 원래 것 그대로, 고정 (Mimir: P_base·d·G_p 불변, ZeroFU: CM·eGen 불변)
    mod_s = s.gp if s.ident == "prompt" else s.cond
    mod_o = orig.gp if orig.ident == "prompt" else orig.cond
    mod_s.load_state_dict(mod_o.state_dict())
    for p in mod_s.parameters():
        p.requires_grad_(False)
    return s


class Unlearner:
    """지우기 한 판의 상태 (학생·생성기·옵티마이저). run(n) 으로 n 번 반복.
    student 를 주면 그 학생에서 이어 간다 (연합 증류에서 라운드마다 서버 평균을 받아 이어 갈 때)."""

    def __init__(self, teacher_r: Net, teacher_f: Net, c_r: torch.Tensor, c_f: torch.Tensor, build, in_ch: int,
                 img: int, cfg: UnlearnCfg, seed: int, student: Net | None = None, total: int | None = None, log=print):
        self.dev = next(teacher_r.parameters()).device
        for m in (teacher_r, teacher_f):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        self.tr, self.tf, self.c_r, self.c_f = teacher_r, teacher_f, c_r, c_f
        self.cfg, self.in_ch, self.img, self.log = cfg, in_ch, img, log
        if student is None:
            self.student = make_student(teacher_r, build, seed)    # 이어서 생성기 초기화 (main 브랜치와 같은 난수 흐름)
        else:
            self.student = student
            torch.manual_seed(seed)
        self.g = Generator(cfg.z_dim, in_ch, img).to(self.dev)
        self.gen_obj = cfg.gen_obj if cfg.gen_obj != "auto" else ("lf_lk" if self.student.ident == "prompt" else "lf")
        total = total or cfg.rounds
        self.opt_s = torch.optim.Adam([p for p in self.student.body_params()], lr=cfg.lr_s)
        self.opt_g = torch.optim.Adam(self.g.parameters(), lr=cfg.lr_g)
        self.sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_s, total)
        self.sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_g, total)
        self.zg = torch.Generator(device=self.dev).manual_seed(seed)
        self.t = 0

    def _losses(self, x):
        cfg = self.cfg
        with torch.no_grad():
            t_out, t_aux = self.tr(x, self.c_r)
            _, f_aux = self.tf(x, self.c_f)
        s_out, s_aux = self.student(x, self.c_r)
        lf = lf_loss(s_aux["feat"], f_aux["feat"]) if cfg.use_lf else s_out.new_zeros(())
        lk = kd_loss(t_out, s_out, cfg.tau)
        la = att_loss(t_aux["acts"], s_aux["acts"])
        return lf, lk, la

    def run(self, n: int) -> Net:
        cfg, dev = self.cfg, self.dev
        lf = lk = la = torch.zeros(())
        for _ in range(n):
            z = torch.randn(cfg.batch_size, cfg.z_dim, device=dev, generator=self.zg)
            if cfg.use_dg:
                self.g.train()
                x = self.g(z).detach()
            else:
                x = torch.randn(cfg.batch_size, self.in_ch, self.img, self.img, device=dev, generator=self.zg)
            self.student.train()
            for _ in range(cfg.kd_steps):
                lf, lk, la = self._losses(x)
                ls = lf + cfg.beta * lk + cfg.gamma * la
                self.opt_s.zero_grad()
                ls.backward()
                self.opt_s.step()
            if cfg.use_dg:
                # 생성기는 학생과 반대로 올린다. 교사는 고정, 학생 가중치 기울기는 버린다.
                xg = self.g(z)
                s_out, s_aux = self.student(xg, self.c_r)
                t_out, _ = self.tr(xg, self.c_r)
                _, f_aux = self.tf(xg, self.c_f)
                lfg = lf_loss(s_aux["feat"], f_aux["feat"])
                lg = lfg + cfg.beta * kd_loss(t_out, s_out, cfg.tau) if self.gen_obj == "lf_lk" else lfg
                self.opt_g.zero_grad()
                (-lg).backward()
                self.opt_g.step()
                self.student.zero_grad(set_to_none=True)
                self.sch_g.step()
            self.sch_s.step()
            self.t += 1
            if self.t % cfg.log_every == 0:
                self.log(f"    unlearn {self.t}  L_F {lf.item():.4f}  L_K {lk.item():.4f}  L_att {la.item():.4f}")
        return self.student


def unlearn(teacher_r: Net, teacher_f: Net, c_r: torch.Tensor, c_f: torch.Tensor, build, in_ch: int, img: int,
            cfg: UnlearnCfg, seed: int, log=print, student: Net | None = None) -> Net:
    """논문 그대로의 한 쌍 지우기. teacher_r 과 teacher_f 는 보통 같은 원래 모델."""
    return Unlearner(teacher_r, teacher_f, c_r, c_f, build, in_ch, img, cfg, seed, student=student, log=log).run(cfg.rounds)
