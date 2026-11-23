#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export VIMMSD_ENV=server
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

plan=configs/experiments/phase11_image_contribution.yaml
output_dir=data/server/outputs
python_bin="${PYTHON_BIN:-python}"

if [[ "${1:-}" == "--dry-run" ]]; then
    "$python_bin" scripts/run_experiments.py --plan "$plan" --dry-run
    exit
fi

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable. Activate the Phase 11 GPU environment."'
"$python_bin" scripts/run_experiments.py --plan "$plan" "$@"

# Check that all three runs are available before creating the comparison bundle.
seeds=(42 1 2)
artifacts=(config.json best_val_metrics.json val_predictions.csv test_predictions.csv run.log)
for seed in "${seeds[@]}"; do
    run="p11_caption_la_s$seed"
    for artifact in "${artifacts[@]}"; do
        file="$output_dir/$run/$artifact"
        if [[ ! -s "$file" ]]; then
            printf 'Missing artifact: %s\n' "$file" >&2
            exit 1
        fi
    done
done

bundle="$(mktemp -d "$output_dir/phase11_caption_la.XXXXXX")"
for seed in "${seeds[@]}"; do
    run="p11_caption_la_s$seed"
    for artifact in "${artifacts[@]}"; do
        cp -- "$output_dir/$run/$artifact" "$bundle/${run}_$artifact"
    done
done
tar -czf "$bundle.tar.gz" -C "$bundle" .
printf 'Comparison bundle: %s.tar.gz\n' "$bundle"
