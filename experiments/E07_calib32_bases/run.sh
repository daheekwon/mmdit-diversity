#!/bin/bash
# E07: new V-edit bases on this server (2026-11-06, user): the separate calibration set splits/calib32.txt
# (32 prompts x seeds 9000000-9000015, data.calibration), all blocks, each model at its config resolution
# (SD3.5 / FLUX.1 1024, klein 512). The earlier bases (bases/*_legacy_dev8*, klein_dev8*) are not on this server.
# One capture per model; several --steps sets give one basis each (bases/<name>_s<steps>_k256.pt).
set -euo pipefail
cd /NHNHOME/workspace/mmdit-diversity
source env.sh
mkdir -p experiments/E07_calib32_bases/logs
L=experiments/E07_calib32_bases/logs
python scripts/fit_basis.py --model sd35  --name sd35_calib32  --steps 0-11 0-3      2>&1 | tee $L/sd35.log
python scripts/fit_basis.py --model flux  --name flux_calib32  --steps 0-5           2>&1 | tee $L/flux.log
python scripts/fit_basis.py --model klein --name klein_calib32 --steps 0-4 0-6 0-10  2>&1 | tee $L/klein.log
