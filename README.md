# mmdit_diversity

이 폴더는 상위 레포(`diversity_migration/`)의 탐색 실험에서 남은 후보 방법만 다시 구현한 것입니다. 정제된 평가 실험도 여기서 합니다.
모델은 Hugging Face diffusers로 불러옵니다(SD3.5 Medium, FLUX.1-dev, 로컬 캐시).
T2V 시험용으로 CogVideoX-2B도 들어 있습니다(`configs/cogvideox.yaml`, `experiments/E02_t2v_cogvideox_pilot/PROTOCOL_ko.md`). 이 모델에서 "image"는 모든 프레임의 영상 토큰을 뜻합니다.
세 번째 이미지 모델로 FLUX.2-klein-base-4B도 들어 있습니다(`configs/klein.yaml`, `experiments/E03_klein_port/PROTOCOL_ko.md`). CFG를 step마다 두 번 호출하므로(조건, 빈 prompt) 모든 성분이 두 호출에 모두 적용됩니다. block 번호는 double 0–4, single 5–24입니다. CADS는 klein에서 아직 지원하지 않습니다.

## 방법

| 성분 | 하는 일 | 설정 키 |
|---|---|---|
| V편집 | joint attention의 image V 행에 `V' = V + (λ−1) Q_b Q_bᵀ (V − b)`를 적용한다. `Q_b`는 block별 seed-대비 기저(`div/basis.py`)이다. | `v_edit.{lam, strength, blocks, steps, basis}` |
| pad V 감쇠 | image query 행이 padding 토큰의 value를 ×factor로 읽는다. softmax 질량과 text query 행은 바꾸지 않는다. | `pad_v.{factor, blocks, steps}` |
| head 조작 (미정) | image query 행의 image key write를 head마다 ×(1 − β(1 − head 선택도))로 줄인다. | `head.{beta, blocks, steps}` |
| DAVE (기준선) | 지정 block에 들어가는 image stream의 토큰 평균(DC)을 ×α로 줄인다(DAVE 논문 표기). | `dave.{alpha, block, steps}` |
| CADS (기준선) | text 조건에 step별 잡음을 넣는다. 잡음과 정규화 통계는 이미지마다 따로 계산한다. | `cads.{noise_scale, tau1, tau2, psi, rescale}` |

- 모든 성분은 이미지 단위로 동작합니다. 배치 통계나 참조 실행을 쓰지 않으므로 배치 구성이 결과에 영향을 주지 않습니다.
- 모든 모델이 같은 코드(`div/method.py`)를 씁니다. 모델별 차이(토큰 배치, block 번호, q/k/v를 가로채는 위치)는 `div/models.py`에만 둡니다.
- V편집 강도는 `lam`(λ)으로 지정합니다. 편집하는 모든 block에 같은 λ가 들어갑니다.
  - `strength`는 다른 지정 방식입니다. image V의 상대 변화량 ‖V′−V‖/‖V‖이고, λ = 1 + strength / unit으로 바뀝니다. unit은 기저 파일에 block별로 저장되며, 편집하는 block들의 평균을 씁니다. CogVideoX 설정이 이 방식을 씁니다.
  - `lam`과 `strength`는 둘 중 하나만 지정해야 합니다. 둘 다 있으면 설정을 읽을 때 오류가 납니다. SD3.5·FLUX에서 `strength`를 쓰려면 `v_edit.lam=null v_edit.strength=0.5`처럼 씁니다.
- 범위 표기는 `all`, `6-23`, `0,2,5-7` 중 하나입니다. block 번호는 SD3.5 0–23, FLUX double 0–18·single 19–56, CogVideoX-2B 0–29, FLUX.2-klein double 0–4·single 5–24입니다.
- **SD3.5·FLUX의 기본 해상도는 1024입니다(사용자 11-05).** `value_diversity/`의 모든 방법 비교와 이전 기저는 512에서 만들었습니다. 512로 돌리려면 `model.resolution=512`를 붙입니다.
- 방법 기본값은 11-05에 정한 설정입니다(`value_diversity/results/generalize/`의 결과, 512에서 정함). pad V는 V편집과 같은 step에 들어갑니다.
  - SD3.5: V편집 λ1.14, 전 block(0–23), step 0–11 + pad V ¼, 전 block, step 0–11. 기저는 `bases/sd35_calib32_s0-11_k256.pt`입니다(E07, 11-06. λ1.14에서 image V 상대 변화는 이전 기저 0.105, 새 기저 0.107).
  - FLUX: V편집 λ1.72, block 19–37, step 0–5 + pad V ¼, 전 block, step 0–5. 기저는 `bases/flux_calib32_s0-5_k256.pt`입니다(E07, 11-06. λ1.72에서 block 19–37의 상대 변화는 이전 0.50, 새 기저 0.516).
  - DAVE 기준 세팅: SD3.5 α0.5(block 6, step 0–3), FLUX α0.2(block 30, step 0–5).
- 비교 규칙(11-04 정정): DAVE 기준 세팅은 DAVE를 돌리는 설정일 뿐입니다. 우리 방법의 강도를 DAVE의 Vendi에 맞추지 않습니다. 우리 방법은 자기 운영점에서 DAVE 기준 세팅과 prompt 단위로 짝지어 비교하고, Vendi와 비용 지표를 나란히 보고합니다(`scripts/compare.py`). 같은 Vendi에서의 보간(`analysis.interpolate`)은 우리 변형끼리 비교할 때만 쓰며, 공통 범위의 여러 수준이나 운영점의 Vendi를 기준으로 합니다.

## 구조

```
configs/   모델 설정, 성분 기본값, preset: sd35.yaml, flux.yaml, klein.yaml (FLUX.2-klein), cogvideox.yaml (영상)
           data.yaml  데이터 경로, 평가 규모
div/       config.py        설정 로드, CLI override, 범위 파싱
           models.py        파이프라인 로드, prompt 인코딩, pad 토큰 위치, 모델별 attention 어댑터
           method.py        Controller: V편집, pad V, head 조작, 기준선 hook 설치, 호출 수 검증
           baselines.py     DAVE, CADS
           basis.py         V편집 기저: 보정 모먼트 수집, 대비 PCA, 강도 단위, 반분 선택도 검증, 이전 기저 변환
           data.py          ImageNet 1000 클래스 / COCO 500 prompt, tune 세트(imagenet_tune, coco_tune), seed 규칙, 실이미지 집합
           splits.py        고정 prompt split(dev, dev8, ho3, coco48)과 그 seed (11-06 새로 만듦)
           features.py      Inception, CLIP, DINOv2, PickScore 특징 (폴더별 캐시)
           metrics.py       이미지 지표 계산
           video_metrics.py 영상 지표 (프레임 일관성, 깜빡임, 판독기, 프레임별 DINO·MUSIQ)
           analysis.py      실행 간 비교 (prompt 단위 paired bootstrap), 우리 변형끼리의 같은 Vendi 보간
scripts/   generate.py, evaluate.py, compare.py, fit_basis.py
           sheet.py         이미지 시트: 나란히 놓은 시트, 조건을 가린 시트와 실패 집계
           lanes.py         공유 GPU에서 작업 배분, OOM 재시도
           evaluate_video.py, video_sheet.py   영상용 평가와 시트
bases/     V편집 기저 (.pt + .json 보고서; 모먼트 _moments.pt는 git 제외)
weights/   LAION aesthetic predictor, Inception 가중치
experiments/<이름>/  PROTOCOL_ko.md (결과 보기 전 작성), ANALYSIS_ko.md
tests/     test_method.py (CPU, 수식 대조), test_klein.py (CPU, klein 어댑터와 split), test_video_metrics.py (CPU, 영상 지표)
           parity.py (GPU, SD3.5·FLUX: 이전 이미지와 픽셀 비교)
           parity_klein.py, parity_cogvideox.py (GPU, diffusers 파이프라인과 픽셀 비교)
outputs/   생성 이미지, 특징, 지표 (git 제외)
env/       Python 환경 명세 (아래 '환경' 절)
```

## 환경

이전 결과는 이전 서버의 conda 환경 `/DATA0/dahee/diversity/.miniconda3/envs/diversity`(Python 3.11.15, CUDA 12.8용 PyTorch 2.11.0)에서 만들었습니다. `env/`는 그 환경을 다시 만드는 명세입니다(2026-11-06 기록).
이 서버(컨테이너)에는 이 명세로 `/NHNHOME/workspace/miniforge3/envs/mmdit-diversity`를 만들었습니다(11-06). 명령 전에 `source env.sh`를 실행합니다(이 환경, `HF_HOME=/NHNHOME/workspace/hf_cache`).

| 파일 | 내용 | 용도 |
|---|---|---|
| `env/environment.yml` | conda의 Python 3.11.15, ffmpeg 6.1.2, 그리고 `requirements-lock.txt` | 새 서버에 같은 환경 만들기 (권장) |
| `env/requirements-lock.txt` | 코드가 쓰는 패키지와 그 의존 패키지 90개. 모두 이전 환경의 버전으로 고정 | conda 없이 pip만 쓸 때 |
| `env/requirements.txt` | 코드가 직접 쓰는 패키지 25개 | 어떤 패키지가 필요한지 볼 때 |
| `env/environment-lock.yml` | 이전 환경 전체의 `conda env export` 결과. 다른 프로젝트의 패키지도 들어 있고, linux-64 빌드로 고정 | 공유 환경을 그대로 복제할 때 |
| `env/eval_models.json` | 이 서버에서 내려받은 평가 모델의 리비전과 sha256 | 같은 평가 모델인지 확인할 때 |

저장소 맨 위 폴더에서 다음을 실행합니다.

```bash
conda env create -f env/environment.yml      # 이 서버: CONDA_PLUGINS_USE_SHARDED_REPODATA=false $MINIFORGE/bin/python -m conda env create -f env/environment.yml -y
source env.sh
python -m pytest -q tests/test_video_metrics.py
```

- conda가 `sqlite3.OperationalError: database is locked`로 멈추면, 앞에 `CONDA_PLUGINS_USE_SHARDED_REPODATA=false`를 붙여 다시 실행합니다.
- 이 서버의 miniforge는 설치 뒤 옮겨져서 `conda` 스크립트의 shebang이 없는 경로를 가리킵니다. `$MINIFORGE/bin/python -m conda`로 실행합니다(`env.sh`).
- 확인한 범위(11-06, 이 서버): 설치된 패키지 버전이 `requirements-lock.txt`와 모두 같습니다. B200(sm_100)에서 GPU 연산이 됩니다.
- 11-06 추가: `openai-clip==1.0.1`, `ftfy==6.3.1`, `wcwidth==0.8.1`. `image-reward`가 선언 없이 `clip`을 import합니다. 버전은 `environment-lock.yml`(이전 환경)의 것입니다.

환경 밖에서 따로 준비하는 것:

- **생성 모델:** SD3.5 Medium, FLUX.1-dev, FLUX.2-klein은 이 서버의 HF 캐시에 설정과 같은 리비전으로 있습니다. CogVideoX-2B는 없습니다.
- **평가 모델:** `python scripts/download_eval_models.py`가 CLIP B/32·L/14, DINOv2-small, PickScore, ImageReward, LPIPS AlexNet을 받고, `weights/`에 FID용 Inception과 aesthetic predictor를 받습니다. 받은 판은 `env/eval_models.json`에 적힙니다.
- **V편집 기저 `bases/*.pt`:** 이 서버에서 새로 적합합니다(`experiments/E07_calib32_bases/run.sh`). 이전 기저(`*_legacy_dev8*`, `klein_dev8*`)는 없습니다. json 보고서만 git에 남아 있습니다.
- **데이터:** `configs/data.yaml`의 ImageNet·COCO·VBench 경로는 `/NHNHOME/workspace/data/` 아래입니다.
- **split:** `dev`, `dev8`, `ho3`는 `div/splits.py`, 보정 세트는 `splits/calib32.txt`에 있습니다(11-06 새로 만듦. `value_diversity/`의 이전 prompt와는 다름). `coco48`은 COCO 테스트·tune 파일이 생긴 뒤 `python scripts/make_coco48.py --captions <captions_val2017.json>`로 만듭니다.
- **옆 폴더 `value_diversity/`:** 이 서버에는 없습니다. `tests/parity.py`와 이전 기저 변환(`fit_basis.py --legacy`), E06의 평가 스크립트에만 필요합니다.
- **옆 폴더 `diversity_anal/`:**
  - 영상 MUSIQ는 `diversity_anal/vendor/iqa_runtime`의 pyiqa 0.1.13과 `diversity_anal/model_cache/torch_eval`의 가중치를 씁니다.
  - 영상 판독기는 `diversity_anal/scripts/judge_flux_breakage.py`(Qwen3-VL)를 씁니다.
  - 경로는 `div/video_metrics.py`의 `ANAL`입니다(이 서버: `/NHNHOME/workspace/diversity_anal/div-attn`. 이 폴더에는 `vendor/iqa_runtime`이 없습니다).

## 사용법

```bash
# 생성: 출력 폴더는 outputs/<model>/<dataset>/<preset__overrides>/
python scripts/generate.py --model sd35 --dataset imagenet --preset ours --gpu 0
python scripts/generate.py --model flux --dataset coco --preset ours v_edit.lam=1.5 pad_v.steps=0-4
python scripts/generate.py --model sd35 --dataset coco --preset dave --n-prompts 50   # 격자 탐색용 소규모

# 평가와 비교
python scripts/evaluate.py outputs/sd35/imagenet/ours outputs/sd35/imagenet/original
python scripts/compare.py outputs/sd35/imagenet/original outputs/sd35/imagenet/ours outputs/sd35/imagenet/dave

# V편집 기저 적합 (보정 prompt: data.calibration = splits/calib32.txt; 실제 명령은 experiments/E07_calib32_bases/run.sh)
python scripts/fit_basis.py --model sd35 --name sd35_calib32 --steps 0-11 0-3   # 기본: 전 block 적합, step 세트마다 기저 하나
```

- preset은 `original`, `ours`, `ours_head`, `v_edit`, `pad_v`, `dave`, `cads`입니다. 그 뒤에 `성분.필드=값`을 여러 개 붙일 수 있습니다.
- 이미 있는 이미지는 건너뜁니다. 그래서 중단된 실행을 다시 시작할 수 있고, `--n-prompts`로 만든 소규모 이미지도 전체 실행에서 그대로 재사용됩니다(같은 prompt는 같은 seed).
- `--shard i/n`으로 prompt를 여러 프로세스에 나눌 수 있습니다.
- 데이터셋은 test용 `imagenet`, `coco`와 tune용 `imagenet_tune`(50 클래스, test와 다른 seed), `coco_tune`(test 500장 밖의 val2017 이미지 50장)입니다. tune 세트의 FID·PRDC는 기반 데이터셋의 실이미지를 씁니다.
- `dev`, `dev8`, `ho3`, `coco48`은 고정 split이고, 각자 seed 목록을 씁니다(`div/splits.py`). 실이미지 집합은 없습니다. 11-06에 새로 만들었습니다. `value_diversity/`의 이전 split과는 prompt가 다릅니다.
  - `coco48`은 `splits/coco48.json`입니다. `scripts/make_coco48.py`가 COCO 테스트 500장, tune 50장과 겹치지 않는 val2017 사진 48장에서 캡션을 하나씩 골랐습니다.
  - V편집 보정 세트(`splits/calib32.txt`, 32개)는 dev·ho3와 겹치지 않습니다.
- 같은 폴더 이름에 다른 설정을 넣으면 오류가 납니다.

## 평가

규모는 DAVE 논문(arXiv 2606.06813)을 따릅니다. ImageNet은 1,000 클래스 × 10장, 실이미지 10K장(클래스당 10장)입니다. COCO는 val2017 이미지 500장에서 캡션을 하나씩 고른 500개 × 10장(COCO 캡션 형식), 실이미지 5K장(val2017 전체)입니다.
  - 파일은 원본 `captions_val2017_img500_cap500.json`과 tune `captions_val2017_tune50.json`입니다(사용자 11-06).
  - 이전 결과는 `_no_coco48` 판을 썼습니다. 이전 coco48과 사진이 같은 5개(tune은 1개)를 다른 사진으로 바꾼 판입니다(`experiments/E01_tune_operating_point/coco_replaced.json`). coco48을 새로 뽑았으므로 이 교체는 더 쓰지 않습니다.

| 단위 | 지표 |
|---|---|
| 이미지 | CLIP score (ViT-B/32, 100×cos), PickScore, Aesthetic (LAION v2), ImageReward |
| prompt | Vendi (CLIP ViT-B/32 커널: 논문과 같음; DINOv2 커널: 이 프로젝트의 주 다양성 지표), 쌍별 LPIPS |
| 실행 전체 | FID, precision, recall, density, coverage (Inception pool3, k=5; 모든 prompt가 완료됐을 때만) |

- 실이미지는 짧은 변을 생성 해상도에 맞추고 가운데를 잘라 생성 이미지와 크기를 맞춥니다. 1024에서는 ImageNet·COCO 실이미지 대부분이 이보다 작아서 키우게 됩니다.
- ImageNet prompt는 `a photo of a {클래스의 첫 동의어}`입니다. 이름이 겹치는 클래스가 두 쌍 있습니다(crane, maillot).
  - 이 서버에는 devkit이 없습니다. 그래서 클래스 이름은 HF imagenet-1k의 `classes.py`에서, 실이미지는 `val/<wnid>/` 폴더에서 읽습니다. HF 목록은 두 번째 crane을 `crane2`로 바꿔 두었으므로 devkit 이름 `crane`으로 되돌립니다(`data.HF_RENAMED`).
- seed 규칙은 `seed_base + 100·i + j`이며 모든 방법에서 같습니다. 그래서 비교는 이미지 단위로 짝지어집니다.
- FID와 PRDC 특징은 clean-fid의 'clean' 방식(TF Inception 가중치 `weights/inception-2015-12-05.pt`, PIL bicubic 299)으로 GPU 하나에서 뽑습니다. clean-fid 자체 함수와의 차이는 GPU 연산 잡음 수준입니다(상대 0.05%, 같은 입력을 두 번 넣어도 이만큼 다릅니다).
- ImageReward 패키지 1.5는 transformers 4.40 이전용입니다. 그래서 `metrics.image_reward`가 옮겨진 함수 세 개를 원래 자리에 되돌린 뒤 불러옵니다. 이렇게 낸 점수는 transformers 4.36에서 낸 점수와 같습니다(이미지 18장 × prompt 3개, 차이 0). 가중치는 `~/.cache/ImageReward`에 있습니다.
- 생성 이미지가 6장 미만이면 PRDC(k=5)는 건너뜁니다.

## 이전 결과와의 관계

- `bases/*_legacy_dev8*_k256.pt`는 `value_diversity/`의 개발 prompt 8개로 적합한 기저입니다. 이전 결과가 모두 이 기저를 썼습니다. 이 서버에는 `.pt`가 없고 json 보고서만 있습니다. 지금 기본값은 E07의 새 기저(`*_calib32_*`)입니다.
  - `sd35_legacy_dev8_k256.pt`: SD3.5 block 6–23. 11-03 운영점과 픽셀 대조에 씁니다.
  - `sd35_legacy_dev8_b0to23_k256.pt`: SD3.5 block 0–23. 전 block 실행(generalize/allblock)이 쓴 기저이고, 6–23 부분은 위 파일과 같습니다. `fit_basis.py --model sd35 --legacy --blocks all`로 만듭니다.
  - `flux_legacy_dev8_k256.pt`: FLUX block 19–56.
- `tests/parity.py`는 같은 prompt와 seed로 이 패키지와 이전 실행기의 이미지를 픽셀 단위로 비교합니다. 모든 비교는 512에서 합니다. 11-03 운영점은 override로 고정해서 비교하고, 지금 기본값도 따로 비교합니다. 결과는 `outputs/parity/<model>.json`에 저장됩니다.
- **`value_diversity/results/generalize/`의 SD3.5 이미지는 hook 방식 실행기로 만들었습니다.** 이 패키지는 이전 실행기의 단일 패스 방식(`--fast`)과 픽셀이 같습니다. hook 방식 이미지와는 픽셀이 다릅니다(11-03 운영점에서 평균 차이 15/255). FLUX는 두 방식이 같습니다.
- **SD3.5 CADS는 이전 결과와 다릅니다.** 이전 구현은 잡음 통계를 배치 전체에서 계산했습니다. 여기서는 이미지마다 계산합니다.

## 미정

- head 조작의 채택 여부
