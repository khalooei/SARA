#!/usr/bin/env bash
# Run every line of a command file, P at a time, logging each run next to its outputs.
# Usage: bash experiments/run_queue.sh commands.txt [P]
CMDS="$1"; P="${2:-12}"
grep -v '^\s*$' "$CMDS" | xargs -P "$P" -I{} bash -c 'cmd="{}"; out=$(echo "$cmd" | sed -E "s/.*--out-dir ([^ ]+).*/\1/"); mkdir -p "$out"; [ -f "$out/final.json" ] && exit 0; rm -f "$out/train_log.jsonl"; eval "$cmd" > "$out/run.log" 2>&1'
