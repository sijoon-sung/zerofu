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

## 실행

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
| 서술자 d_i 초기값, P_base 초기값, l_k·l_v | N(0,1), 1 근처, 16 | 없음 |
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

---

## 주제 A (브랜치 `topic/continue-after-unlearning`): 지운 뒤에도 연합학습은 계속된다

**질문.** 두 논문의 지우기는 남는 클라이언트 C_r 한 명과 지울 클라이언트 C_f 한 쌍 사이에서 일어나고,
결과 학생은 C_r 에서만 쓰인다.

- Mimir §IV-C: "the unlearned model continues to work on C_r"
- Mimir §VI-G: 지우기 시간은 C_r 과 C_f 사이 계산이라 클라이언트 수와 무관
- ZeroFU §3.1: 지우기는 클라이언트 쪽에서, 교사는 C_r 의 모델

그러면 나머지 클라이언트는 C_f 가 섞인 원래 모델을 그대로 갖고, 연합은 그 상태로 다시 이어진다.
그렇다고 모두가 각자 학생을 만들면 학생마다 무작위로 처음부터 배워서 평균할 수 없다.
**데이터 없이 지운 뒤 연합 전체가 다시 함께 학습하려면 서버는 무엇으로 다시 시작해야 하나?** 두 논문 모두 다루지 않는다.

**실험.** 지운 직후 서버가 갖는 모델(프로토콜)을 바꿔 가며, 남은 클라이언트로 연합학습을 C 라운드 더 돌리고
라운드마다 C_f 데이터 정확도·백도어 성공률(되살아나는가)과 남은 클라이언트 전원의 정확도(다른 사람이 망가졌나)를 잰다.

| 프로토콜 | 서버가 다시 시작하는 모델 |
|---|---|
| `none` | 지우기 없이 원래 모델 (C_f 만 빠짐 = 그냥 떠나기, 미세조정 기준선) |
| `pair_adopt` | 논문 그대로 한 쌍 지우기의 학생을 전역 모델로 채택 |
| `all_pairs_avg` | 남은 전원이 각자 한 쌍 지우기 (각자 무작위 초기화, 논문처럼) → 평균 |
| `all_pairs_shared` | 같지만 학생 초기값을 모두 같게 → 평균 |
| `fed_distill` | 같은 초기 학생에서 출발, 라운드마다 각자 자기 교사로 조금씩 지우기 증류 → 서버 평균 (총 반복 수는 한 쌍 지우기와 같음) |
| `retrain` | C_f 없이 처음부터 학습한 모델 (기준) |

```bash
python run.py --config configs/mimir.toml --stage continue
python run.py --config configs/zerofu.toml --stage continue
python plot_continue.py --tag mimir_mnist_z001          # results/<tag>/fig_continue.png
```

결과: `results/<tag>/continue.csv` (protocol, t, df_max, df_r, asr_max, ret_mean, ret_min, cr_acc, test_mean).
`all_pairs_*` 는 한 쌍 지우기를 남은 인원수만큼 하므로 비용이 그만큼 든다 (로그에 시작 모델 만드는 데 걸린 시간).

**미리 보는 예상 (검증 전).** `pair_adopt` 는 학생이 C_r 교사에게서만 배워서 다른 남은 클라이언트 정확도(ret_min)가
처음에 크게 떨어질 수 있고, 이어지는 학습이 그걸 회복하는 동안 C_f 흔적이 되살아나는지가 관심사다.
`all_pairs_avg` 는 평균이 깨질 것으로 예상하고, `all_pairs_shared`·`fed_distill` 이 그걸 고치는지 본다.
