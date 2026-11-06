# source this before any command in this repo:  source env.sh
#
# Container paths (2026-11-06). The repo used to live at /DATA0/dahee/diversity_migration/mmdit_diversity.
# Python: the conda env built from env/environment.yml (README "환경"); this miniforge was moved after install, so its
# `conda` script has a stale shebang: run conda as `$MINIFORGE/bin/python -m conda ...`.
# Models come from the shared HF cache (SD3.5 and FLUX configs have no cache_dir, so HF_HOME must point there or the
# local_files_only loads fail).
export DIV_ROOT=/NHNHOME/workspace/mmdit-diversity
export MINIFORGE=/NHNHOME/workspace/miniforge3
export CONDA_PREFIX=$MINIFORGE/envs/mmdit-diversity
export PATH=$CONDA_PREFIX/bin:$PATH
export HF_HOME=/NHNHOME/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
