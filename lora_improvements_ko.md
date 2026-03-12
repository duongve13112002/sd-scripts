# LoRA 학습 정밀도 개선 사항

## 개요

이 문서는 Anima/FLUX LoRA 학습 파이프라인에 추가된 두 가지 정밀도 관련 기능인 **`lora_fp32_accumulation`**과 **`attn_softmax_scale`**을 설명한다. 두 기능 모두 bfloat16 학습에서 발생하는 수치 정밀도 손실이라는 공통된 문제를 해결하지만, forward pass의 서로 다른 단계에서 작동하며 위험/보상 프로필이 다르다.

---

## 문제: bf16 LoRA 학습에서의 정밀도 손실

bfloat16은 유효 자릿수가 약 3자리에 불과하다 (가수부 7.8비트 vs. fp32의 23비트). 이로 인해 두 가지 문제가 발생한다:

1. **LoRA 델타 소실**: LoRA의 기여분 (`down → up` 행렬곱)은 기본 모델 출력 대비 작은 델타를 생성한다. bf16에서 이 델타를 더하면 양자화로 인해 0으로 반올림될 수 있으며, 해당 forward pass에서 LoRA 레이어가 사실상 아무 일도 하지 않게 된다.

2. **Softmax 포화**: 어텐션 로짓 (Q·K에 1/√d를 곱한 값)이 좁은 수치 범위로 압축되면, bf16 softmax가 토큰 간 차이를 구별하지 못한다. 어텐션 분포가 거의 균등해지면서 모델의 집중 능력이 상실된다.

이 문제들은 복합적으로 작용한다: 부정확한 어텐션 → 잡음이 많은 그래디언트 → 부정확한 LoRA 업데이트 → 수렴 저하.

---

## 기능 1: `lora_fp32_accumulation`

**플래그:** `--lora_fp32_accumulation`
**파일:** `networks/lora_flux.py` (구현), `networks/lora_anima.py` (상속), `library/anima_train_utils.py` (인자 정의)

### 동작 방식

LoRA의 down→up 행렬곱을 fp32로 업캐스트하고, fp32에서 스케일링을 적용한 후, 기본 출력에 더하기 전에 모델의 원래 dtype으로 다시 캐스트한다.

```python
# 기본 경로 (전체 bf16)
lx = self.lora_down(x)                    # bf16 행렬곱
lx = self.lora_up(lx)                     # bf16 행렬곱
return org_forwarded + lx * multiplier     # bf16 덧셈 — 델타가 소실될 수 있음

# FP32 누적 경로
lx = F.linear(x.float(), self.lora_down.weight.float())   # fp32 행렬곱
lx = F.linear(lx, self.lora_up.weight.float())            # fp32 행렬곱
lx = (lx * multiplier * scale).to(org_forwarded.dtype)    # fp32에서 스케일링 후 캐스트
return org_forwarded + lx                                  # bf16에서 덧셈
```

### 왜 중요한가

hidden_dim=3072인 선형 레이어에 rank-4 LoRA를 적용하는 경우를 생각해보자. 델타는 (3072×4)와 (4×3072) 행렬의 곱이며, 4차원 병목을 통과하기 때문에 본질적으로 값이 작다. bf16에서는 기본 활성화 값 대비 ~1e-2보다 작은 값이 반올림으로 사라진다.

FP32 누적은 이러한 델타를 약 7자리의 유효 자릿수로 보존하여, LoRA의 기여분이 실제로 잔차 스트림에 도달하도록 보장한다.

### 비용

| 항목 | 영향 |
|------|------|
| VRAM | 무시할 수준. 임시 fp32 텐서는 rank 크기 (예: 4×3072)이며 모델 크기가 아님 |
| 속도 | 거의 없음. fp32 행렬곱은 작은 행렬에 대해 수행되며, 기본 모델의 forward/backward가 지배적 |
| 정확성 | 엄밀히 더 정확함 — 동작 변경 없이 정밀도만 향상 |

### 권장 사항

**항상 활성화할 것.** 단점이 없다. 정밀도 향상은 낮은 rank (≤32)와 bf16 학습에서 가장 효과적이지만, 높은 rank에서도 비용이 0이므로 끌 이유가 없다.

---

## 기능 2: `attn_softmax_scale`

**플래그:** `--attn_softmax_scale <float>`
**파일:** `library/attention.py` (모든 백엔드에 전파), `library/anima_models.py` (모델 통합), `library/anima_train_utils.py` (인자 정의)

### 동작 방식

기본 어텐션 스케일 팩터 (1/√head_dim)를 사용자 정의 값으로 재정의한다. 이 스케일은 softmax 전에 Q·K 내적에 적용되며, 지원되는 모든 백엔드에서 동작한다:

| 백엔드 | 파라미터 |
|--------|----------|
| PyTorch SDPA | `scale=` |
| xFormers | `scale=` |
| SageAttention | `sm_scale=` |
| Flash Attention | `softmax_scale=` |

### 왜 중요한가

head_dim=128일 때, 기본 스케일은 1/√128 ≈ 0.088이다. 이는 어텐션 로짓을 좁은 범위로 압축한다. bf16에서:

- **작은 스케일 (기본값 0.088):** 로짓이 0 근처에 밀집 → softmax 출력이 거의 균등 → 어텐션이 토큰을 구별하지 못함
- **큰 스케일 (예: 0.12):** 로짓이 더 넓게 분포 → softmax가 더 날카로운 분포 생성 → bf16에서 더 나은 토큰 구별

이는 저정밀도 학습 불안정성에 대한 연구에 기반하며, 어텐션 softmax가 정밀도 감소 시 가장 먼저 열화되는 구성 요소임을 보여준다.

### 비용

| 항목 | 영향 |
|------|------|
| VRAM | 없음 |
| 속도 | 없음 (스칼라 곱셈 한 번) |
| 정확성 | **모델 동작이 변경됨.** 스케일이 클수록 어텐션이 날카로워짐. 너무 크면 발산할 수 있음 |

### 주의 사항

- **Flash Attention은 내부적으로 이미 softmax를 fp32로 누적한다.** `--attn_mode flash`를 사용 중이라면, 사용자 정의 스케일의 정밀도 이점은 줄어든다 (날카로움 효과는 유지됨).
- **이것은 하이퍼파라미터이지, 공짜 점심이 아니다.** 잘못된 값은 수렴을 해칠 수 있다. head_dim=128에서 권장 범위는 0.10–0.15이다.
- **추론 불일치 위험:** 기본값이 아닌 스케일로 학습하면, 추론 시에도 동일한 스케일을 사용해야 일관된 결과를 얻을 수 있다.

### 권장 사항

**비-Flash 백엔드 (torch, sageattn)와 bf16 학습에서 실험할 가치가 있다.** 0.10부터 시작하여 기본값과 loss 곡선을 비교하라. 이미 Flash Attention을 사용 중이라면 추가 이점은 작다.

---

## 기존 학습 스크립트와의 비교

### 기능 지원 현황

| 기능 | `lora.py` (SD1/2) | `lora_flux.py` (FLUX) | `lora_anima.py` (Anima) |
|------|:---:|:---:|:---:|
| FP32 누적 | - | **Yes** | **Yes** (상속) |
| Softmax 스케일 | - | - | **Yes** |
| Split QKV dims | - | **Yes** | **Yes** (상속) |
| GGPO | - | **Yes** | **Yes** (상속) |
| Rank dropout | Yes | Yes | Yes |
| Module dropout | Yes | Yes | Yes |
| LoRA+ | Yes | Yes | Yes |
| 정규식 기반 LR | - | - | **Yes** |

### 학습 스크립트별 정밀도 처리

| 기능 | `train_network.py` | `sdxl_train_network.py` | `flux_train_network.py` | `anima_train_network.py` |
|------|:---:|:---:|:---:|:---:|
| mixed_precision (fp16/bf16) | Yes | Yes | Yes | Yes |
| full_fp16 / full_bf16 | - | Yes | Yes | Yes |
| FP8 기본 모델 | - | - | Yes | - |
| LoRA fp32 누적 | - | - | - | **Yes** |
| 어텐션 softmax 스케일 | - | - | - | **Yes** |
| Unsloth 오프로드 체크포인팅 | - | - | - | **Yes** |

### 기존 스크립트에 없는 것들

기본 `train_network.py`와 `sdxl_train_network.py`는 정밀도를 고려한 LoRA 연산이 없다 — forward pass가 전적으로 모델의 dtype으로 실행된다. fp32 학습에서는 문제없지만, bf16에서는 눈에 보이지 않는 품질 저하를 초래한다:

- **FP32 누적 없음:** LoRA 델타가 bf16에서 계산되고 더해진다. 낮은 rank에서는 반올림 과정에서 기여분이 사실상 폐기된다.
- **Softmax 스케일 제어 없음:** 1/√head_dim에 고정되어 있다. bf16의 제한된 동적 범위를 보상할 방법이 없다.
- **GGPO 없음:** 견고성을 위한 그래디언트 기반 교란이 없다. 기존 LoRA는 정규화를 위해 전적으로 dropout에 의존한다.

`flux_train_network.py`는 FP8을 지원하고 `lora_flux.py` 모듈은 fp32 누적과 GGPO를 지원하지만, 학습 스크립트가 `--lora_fp32_accumulation`이나 `--attn_softmax_scale`을 인자로 노출하지 않는다 — 이 기능들은 Anima 학습 경로에서만 연결되어 있다.

---

## 실전 가이드

### 상황별 사용법

| 상황 | fp32 누적 | softmax 스케일 | 비고 |
|------|:-:|:-:|------|
| bf16 + 낮은 rank (≤32) | **Yes** | 0.10–0.12 시도 | 최대 효과 — 작은 델타가 bf16 반올림에 가장 취약 |
| bf16 + 높은 rank (≥64) | Yes (무료) | 선택 사항 | 델타가 크므로 소실 가능성이 낮지만, 활성화 비용도 없음 |
| bf16 + Flash Attention | **Yes** | 낮은 우선순위 | Flash가 이미 내부적으로 fp32 softmax 수행 |
| bf16 + SDPA/sageattn | **Yes** | **Yes** | 두 정밀도 수정이 상호 보완적 |
| fp32 학습 | 불필요 | 불필요 | 전체 정밀도이므로 반올림 문제 없음 |
| fp16 학습 | **Yes** | 0.10–0.12 시도 | fp16은 bf16보다 가수부 비트가 많지만 여전히 효과 있음 |

### 사용 예시

```bash
python anima_train_network.py \
    --mixed_precision bf16 \
    --full_bf16 \
    --lora_fp32_accumulation \
    --attn_softmax_scale 0.11 \
    --attn_mode sageattn \
    --network_dim 16 \
    ...
```

---

## 요약

| | `lora_fp32_accumulation` | `attn_softmax_scale` |
|---|---|---|
| **메커니즘** | LoRA 행렬곱을 fp32로 업캐스트 | 어텐션 로짓 범위를 확장 |
| **유형** | 정밀도 수정 (투명) | 하이퍼파라미터 (동작 변경) |
| **비용** | ~0 | 0 |
| **위험** | 없음 | 너무 크면 수렴에 영향 |
| **가장 유용한 경우** | 낮은 rank LoRA + bf16 | 비-Flash 어텐션 + bf16 |
| **결론** | 항상 활성화 | 신중하게 실험 |

`lora_fp32_accumulation`은 트레이드오프가 없는 순수한 개선이므로 기본적으로 활성화해야 한다. `attn_softmax_scale`은 bf16 어텐션에서 더 많은 정밀도를 확보하기 위한 유용한 조절 수단이지만, 튜닝이 필요하며 Flash Attention의 내장 fp32 softmax와 이점이 중복된다.
