# zeroshot_fu — 데이터 없는 개인화 연합 언러닝 재현 (Mimir · ZeroFU)

같은 연구팀의 두 논문을 한 코드에서 재현한다. 두 논문은 학습·지우기 절차가 같고 **클라이언트 이름표(조건)만 다르다.**

| | Mimir (IEEE TMC 2025, 10.1109/TMC.2025.3570018) | ZeroFU (IJCAI 2025, 10.24963/ijcai.2025/733) |
|---|---|---|
| 이름표 | 클라이언트마다 학습되는 서술자 d_i → 프롬프트 P_i (식 7, 8) | 라벨 분포 LD_i → 조건 모듈 CM (식 4~11, GPFL 방식) |
| 지우기 | GAN + 증류, 생성기 목표 L_F + βL_K (식 19) | 같은 구조, 생성기 목표 L_F (식 18) |
| 설정 | 2층 CNN, MNIST·SVHN·FMNIST·CIFAR-10, 10명, Dir(0.01/0.1) | 같음 |

설정에서 `model.ident = "prompt"` 이면 Mimir, `"labeldist"` 이면 ZeroFU 다.

## 구조

```
zfu/data.py        데이터·분할(dirichlet/pathological)·백도어(C_f 최다 클래스에 3×3 흰 사각형, 라벨 y+1)
zfu/model.py       백본 CNN, PromptGen(Mimir), LabelCond(ZeroFU), 생성기
zfu/fl.py          개인화 연합학습 (Mimir 알고리즘 1 / ZeroFU 알고리즘 1), 라운드마다 저장·재개
zfu/unlearn.py     데이터 없는 지우기 (알고리즘 2): 교사 T_R, 지울 교사 T_F, 무작위 학생 S, 생성기 DG
zfu/metrics.py     Dr/Df 정확도, 백도어 ASR, 멤버십 추론, 재학습과의 가중치 거리
zfu/experiment.py  단계: pretrain → retrain → unlearn(pair) / sequential(연속 삭제 틀)
run.py             진입점
configs/           mimir.toml, zerofu.toml (논문 값은 출처 주석, 없는 값은 "가정")
```

## 한 번에 실행 (딸칵)

```bash
pip install -r requirements.txt     # Python 3.11 이상 (tomllib), CUDA 판 torch 권장
run_all.bat                         # 윈도우: 더블클릭. 리눅스/맥: python run_all.py
run_all.bat --estimate              # 이 컴퓨터에서 몇 시간 걸릴지 1~2분 재고 끝
```

- 두 방법 × 4 데이터(MNIST·FMNIST·SVHN·CIFAR-10) × ζ(0.01, 0.1) × 논문의 (C_r, C_f) 2쌍 = **작업 32개**.
  SVHN ζ=0.1 에서는 논문 절제(L_F 뺌, 생성기 뺌)도 같이.
- 데이터는 `data/` 에 자동으로 받는다. 끊겨도 다시 실행하면 끝난 작업은 건너뛰고 하던 작업은 라운드 단위로 이어 간다.
- 끝나면 `results/summary.md` 에 **논문 표(Mimir 표 III, ZeroFU 표 1) 값과 우리 값을 나란히** 적는다.
- 옵션: `--workers 3` (동시 작업 수, 기본 2), `--only mimir`, `--datasets mnist cifar10`, `--zetas 0.1`, `--rounds 50`, `--summary` (요약만), `--smoke` (배관 확인).

## 하나씩 실행

```bash
python run.py --config configs/mimir.toml --stage pair --smoke        # 축소 설정, 1~2분
python run.py --config configs/mimir.toml --stage pair                # 표 III 한 칸 (C_r=0, C_f=1, MNIST, ζ=0.01)
python run.py --config configs/zerofu.toml --stage pair --set data.zeta=0.1 --set pair.retained=4 --set pair.forget=5 --tag zerofu_mnist_z01
python run.py --config configs/mimir.toml --stage sequential          # 연속 삭제 (논문에 없는 후속 실험 틀)
```

`--set 구역.키=값` 으로 무엇이든 바꾼다 (값은 TOML 문법). 산출물이 이미 있으면 그 단계는 건너뛴다.
결과: `results/<tag>/results.csv`, `sequential.csv`. 로그: `logs/<tag>.log`.

## 논문에 없어서 정한 것 (재현 시 차이의 원인이 될 수 있음)

| 항목 | 정한 값 | 근거 |
|---|---|---|
| 연합학습 라운드 수 | 20 | Mimir 그림 1 예시 |
| SGD 모멘텀 | 0 | 식 (11) 이 순수 경사하강 |
| Mimir λ_g 근접항 (식 1) | 뺌 | 알고리즘 1 에 없음 |
| 서술자 d_i, P_base, 프롬프트 생성기 투영 초기값, l_k·l_v | 전부 N(0,1), 16 | 없음. 식 (7) 을 글자 그대로 옮기면 특징 차원 하나가 토큰 하나라, P_base 를 1 근처로 두거나 투영을 작게(또는 0) 두면 모든 클라이언트 프롬프트가 같아져 개인화가 안 생긴다 (확인함) |
| 정규화 λ‖·‖² | 원소 **평균** (`fl.reg = "mean"`) | 식 (12)(14) 그대로 원소 합으로 하면 λ=0.1 이 P_base 를 0 으로 끌어 프롬프트가 모두 같아진다 (10 라운드 뒤 클라이언트 간 분산 1e-6, 확인함). 평균으로 하면 클라이언트별 정확도 98~100% 로 논문 수준 |
| 지우기 반복 T_u, 가짜 배치, 옵티마이저 | 2000, 128, Adam 2e-3/1e-3 + 코사인 | ZSKT(Micaelli & Storkey 2019) 기본값 |
| L_K 의 교사 | T_R | Mimir 식 (16) 아래 문장이 T 를 T_F 로 적었지만 "T_R 과 S 사이 지식 전달"이라 함 |
| 학생 초기화 | 무작위, 이름표 부품은 복사해 고정 | ZeroFU 알고리즘 2 줄 1-2 |
| Df 정확도의 이름표 | C_r 것 (지운 모델은 C_r 에서 계속 쓰임) + 남은 전원 중 최댓값 | 논문은 명시 안 함 |
| 멤버십 추론 평가 집합 | C_f 데이터 + 같은 수의 시험셋 | 논문은 공격 모델 학습만 서술 |
| 백도어 비율 | C_f 최다 클래스의 30% | 없음 |

## 지표

- `dr_acc`, `df_acc`: C_r·C_f 학습 데이터 정확도. 재학습(`retrain`)과 가까울수록 좋음.
- `df_acc_max`: C_f 데이터를 남은 클라이언트 이름표 전부로 재서 가장 높은 값.
- `asr`: 백도어 성공률. `mia_prec/rec`: 멤버십 추론. `wdist`: 재학습과의 θ·φ 거리.

## 지금까지 알게 된 것 (MNIST ζ=0.01, C_r=0, C_f=1)

- 정규화를 식 그대로(원소 합) 걸면 두 방법 모두 개인화가 안 생기고, 그 상태에서는 지우기도 안 된다
  (Mimir: 지운 뒤 Df 94%, 재학습 2%). 평균으로 바꾸면 개인화가 논문 수준으로 생긴다.
- 개인화가 생기면 **지우기 전 원래 모델에 C_r 이름표만 붙여도 C_f 데이터를 0% 맞힌다.**
  논문 표의 "지운 뒤 Df ≈ 0" 은 상당 부분 개인화에서 온다. 요약표의 "Origin(C_r 이름표) Df" 칸이 이 기준선이다.
- ZeroFU 생성기 목표를 식 (18) 그대로(L_F 만) 쓰면 학생이 C_r 데이터도 못 맞힌다 (Dr 0%).
  변형 `gen_lflk` (Mimir 식 19 처럼 L_F + βL_K) 로 비교할 수 있다.
