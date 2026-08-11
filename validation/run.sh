#!/usr/bin/env bash
set -euo pipefail

validation_dir=$(cd "$(dirname "$0")" && pwd)
crate_dir=$(cd "$validation_dir/.." && pwd)
legacy_dir="$validation_dir/legacy-source"

if [[ ! -f "$legacy_dir/Cargo.toml" ]]; then
    mkdir -p "$legacy_dir"
    git -C "$crate_dir" archive legacy:src | tar -x -C "$legacy_dir"
fi
sed -i '0,/name = "general-lotka-volterra-rs"/s//name = "general-lotka-volterra-legacy"/' \
    "$legacy_dir/Cargo.toml"

run_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
run_dir="$validation_dir/runs/$run_id"
mkdir -p "$run_dir"

for example in \
    mean_field_replicator \
    mean_field_replicator_demographic \
    spatial_replicator \
    spatial_general_lotka_volterra \
    ground_truth_comparison
do
    cargo run --manifest-path "$crate_dir/Cargo.toml" --example "$example" \
        >"$run_dir/$example.log" 2>&1
done

recording=$(find "$crate_dir/examples/mean_field_replicator/output" \
    -mindepth 2 -maxdepth 2 -type d -name 'task-000000' -printf '%T@ %p\n' \
    | sort -n | tail -1 | cut -d' ' -f2-)
python "$crate_dir/tools/plot_workflow_recording.py" "$recording" \
    --csv "$run_dir/mean-field-signal.csv"

cargo run --manifest-path "$validation_dir/Cargo.toml" -- "$run_dir" \
    >"$run_dir/legacy-comparison.log" 2>&1

echo "$run_dir"
