#!/bin/bash
set -e

# gpt-oss-20b: zeroshot 0002-0009, then ruleset 0001-0010

MODEL="gpt-oss-20b"

echo "=== Phase 1: zeroshot ==="
for i in $(seq -w 1 3); do
    id="000${i}"
    echo "--- ${id} ---"
    uv run patholint single -r "$id" -m "$MODEL"
done

echo ""
echo "=== Phase 2: ruleset ==="
for i in $(seq -w 1 3); do
    id="000${i}"
    echo "--- ${id} ---"
    uv run patholint single -r "$id" -m "$MODEL" --ruleset
done

echo ""
echo "=== Done ==="
