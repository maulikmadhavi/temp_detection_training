#!/bin/bash
# vitg.vitgyolov8 YOLO 🚀, GPL-3.0 license
# Download latest models from https://github.com/vitg.vitgyolov8/assets/releases
# Example usage: bash vitg.vitgyolov8/yolo/data/scripts/download_weights.sh
# parent
# └── weights
#     ├── yolov8n.pt  ← downloads here
#     ├── yolov8s.pt
#     └── ...

python - <<EOF
from vitg.network.backbone.vitgyolov8.yolo.utils.downloads import attempt_download_asset

assets = [f'yolov8{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '-cls', '-seg')]
for x in assets:
    attempt_download_asset(f'weights/{x}')

EOF
