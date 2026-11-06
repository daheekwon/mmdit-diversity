#!/bin/bash
# 11-05 13:10, request from another session: the sx_coco queue's SD3.5 judge job U4TJ needs 35 GB free on one GPU.
# Keeps new klein generation jobs off GPU 0 (outputs/.pause_gpu0; scripts/generate.py waits while it exists, before
# touching the GPU) until that queue logs "END U4TJ", 4 h at most. Running klein jobs are not touched.
set -u
cd /NHNHOME/workspace/mmdit-diversity
L=experiments/E03_klein_port/logs
Q=/NHNHOME/workspace/value_diversity/results/generalize/sx_coco/queue/nohup.out
P=outputs/.pause_gpu0
log() { echo "$(date '+%F %T') gpu0 hold: $*" >> $L/driver.log; }
touch $P
log "on (sx_coco U4TJ needs GPU 0)"
end=$(( $(date +%s) + 4*3600 ))
until grep -q "END U4TJ" $Q || [ $(date +%s) -ge $end ]; do sleep 60; done
rm -f $P
log "off ($(grep -q 'END U4TJ' $Q && echo 'END U4TJ seen' || echo '4 h timeout'))"
