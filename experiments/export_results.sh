#!/bin/bash
# Copy the configuration, per-epoch log, console log and final metrics of every run (no model weights)
# from runs/ to a results directory, and regenerate the LaTeX rows of Tables 16-18 next to them.
#   bash experiments/export_results.sh results/revision2
set -e
DEST="${1:-results/revision2}"
mkdir -p "$DEST"
for group in main tune pilot; do
  [ -d "runs/$group" ] || continue
  rsync -a --prune-empty-dirs --include='*/' \
    --include='config.json' --include='train_log.jsonl' --include='final.json' --include='run.log' \
    --exclude='*' "runs/$group/" "$DEST/$group/"
done
python experiments/make_tables.py "$DEST/tables" > /dev/null
python experiments/paired_stats.py runs/main > "$DEST/paired_stats.txt"
echo "exported $(find "$DEST" -name final.json | wc -l) runs to $DEST"
