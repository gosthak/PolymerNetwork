#!/bin/bash
# run_all.sh — Launch the full CATCHY pipeline
#
# Usage:
#   bash run_all.sh --config ../configs/default.yaml --platform CUDA
#   bash run_all.sh --config ../configs/default.yaml --platform CPU

set -e

CONFIG="../configs/default.yaml"
PLATFORM="CUDA"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)   CONFIG="$2";   shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Inject platform into a temp config if needed
TMPCONFIG=$(mktemp /tmp/catchy_cfg_XXXX.yaml)
python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
cfg['simulation']['platform'] = '$PLATFORM'
yaml.dump(cfg, open('$TMPCONFIG','w'))
"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==========================================================="
echo "  CATCHY full pipeline"
echo "  Config:   $CONFIG"
echo "  Platform: $PLATFORM"
echo "  Start:    $(date)"
echo "==========================================================="

echo ""
echo "--- Step 1: Build and equilibrate network ---"
python3 01_build_network.py --config "$TMPCONFIG"

echo ""
echo "--- Step 2: Embed enzymes ---"
python3 02_embed_enzymes.py --config "$TMPCONFIG"

echo ""
echo "--- Step 3: Production runs ---"
python3 03_production.py --config "$TMPCONFIG" --mode both

echo ""
echo "--- Analysis: generate all figures ---"
cd ../analysis
python3 plot_all.py --config "$TMPCONFIG"

echo ""
echo "==========================================================="
echo "  DONE: $(date)"
echo "  Figures in: $(python3 -c "import yaml; cfg=yaml.safe_load(open('$TMPCONFIG')); print(cfg['output']['dir'])")/figs/"
echo "==========================================================="

rm -f "$TMPCONFIG"
