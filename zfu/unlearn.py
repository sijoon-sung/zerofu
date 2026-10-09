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


def unlearn(teacher_r: Net, teacher_f: Net, c_r: torch.Tensor, c_f: torch.Tensor, build, in_ch: int, img: int,
            cfg: UnlearnCfg, seed: int, log=print) -> Net:
    """teacher_r 과 teacher_f 는 보통 같은 원래 모델. 연속 삭제 실험에서는 서로 다를 수 있다."""
    dev = next(teacher_r.parameters()).device
    for m in (teacher_r, teacher_f):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    student = make_student(teacher_r, build, seed)
    g = Generator(cfg.z_dim, in_ch, img).to(dev)
    gen_obj = cfg.gen_obj if cfg.gen_obj != "auto" else ("lf_lk" if student.ident == "prompt" else "lf")
    opt_s = torch.optim.Adam(student.body_params(), lr=cfg.lr_s)
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg.lr_g)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, cfg.rounds)
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, cfg.rounds)
    zg = torch.Generator(device=dev).manual_seed(seed)

    def losses(x, need_att: bool):
        with torch.no_grad():
            t_out, t_aux = teacher_r(x, c_r)
            _, f_aux = teacher_f(x, c_f)
        s_out, s_aux = student(x, c_r)
        lf = lf_loss(s_aux["feat"], f_aux["feat"]) if cfg.use_lf else s_out.new_zeros(())
        lk = kd_loss(t_out, s_out, cfg.tau)
        la = att_loss(t_aux["acts"], s_aux["acts"]) if need_att else s_out.new_zeros(())
        return lf, lk, la

    for t in range(cfg.rounds):
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=dev, generator=zg)
        if cfg.use_dg:
            g.train()
            x = g(z).detach()
        else:
            x = torch.randn(cfg.batch_size, in_ch, img, img, device=dev, generator=zg)
        student.train()
        for _ in range(cfg.kd_steps):
            lf, lk, la = losses(x, True)
            ls = lf + cfg.beta * lk + cfg.gamma * la
            opt_s.zero_grad()
            ls.backward()
            opt_s.step()
        if cfg.use_dg:
            # 생성기는 학생과 반대로 올린다. 교사는 고정, 학생 가중치 기울기는 버린다.
            xg = g(z)
            s_out, s_aux = student(xg, c_r)
            t_out, _ = teacher_r(xg, c_r)
            _, f_aux = teacher_f(xg, c_f)
            lf = lf_loss(s_aux["feat"], f_aux["feat"])
            lg = lf + cfg.beta * kd_loss(t_out, s_out, cfg.tau) if gen_obj == "lf_lk" else lf
            opt_g.zero_grad()
            (-lg).backward()
            opt_g.step()
            student.zero_grad(set_to_none=True)
            sch_g.step()
        sch_s.step()
        if (t + 1) % cfg.log_every == 0:
            log(f"    unlearn {t + 1}/{cfg.rounds}  L_F {lf.item():.4f}  L_K {lk.item():.4f}  L_att {la.item():.4f}")
    return student
