# experiments

실험 하나에 폴더 하나를 씁니다. 폴더 이름 예: `E01_vpad_sweep_imagenet`.

- `PROTOCOL_ko.md`는 결과를 보기 전에 씁니다. 여기에 가설, 조건(preset과 override), 데이터 규모(tune 또는 전체), 판정 규칙을 적습니다.
- `run.sh`에는 실제로 실행한 `scripts/generate.py`·`evaluate.py` 명령을 그대로 남깁니다.
- `ANALYSIS_ko.md`에는 결과 표(`scripts/compare.py` 출력), 이미지 확인 결과, 판정을 적습니다.

이미지와 지표 자체는 `outputs/`에 있습니다. 실험 폴더에는 명령과 해석만 둡니다.
