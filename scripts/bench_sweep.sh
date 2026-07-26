#!/usr/bin/env bash
# Context-depth benchmark sweep: prefill / decode / TTFT vs prompt length.
set -uo pipefail

MODEL="${1:-./models/Llama-3.2-3B-Instruct-quantized.w8a8}"
BIN=./build/runtherder
RUNS=3
PARA="The history of computing spans many centuries and involves countless inventors, mathematicians, and engineers who each contributed pieces to the puzzle of automated calculation and information processing across generations. "
REPEATS=(1 25 70 140 300)

PROMPT_FILE="$(mktemp)"
trap 'rm -f "$PROMPT_FILE"' EXIT

printf "%-12s %-16s %-16s %-12s\n" "prompt_tok" "prefill(tok/s)" "decode(tok/s)" "TTFT(s)"
printf -- "------------------------------------------------------------\n"

for rep in "${REPEATS[@]}"; do
    python3 -c "import sys; sys.stdout.write('${PARA}' * ${rep})" > "$PROMPT_FILE"
    ptok=""; pf_sum=0; dc_sum=0; ttft_sum=0; n=0
    for r in $(seq 1 "$RUNS"); do
        out="$("$BIN" "$MODEL" --temperature 0.0 "$(cat "$PROMPT_FILE")" 2>/dev/null)"
        ptok="$(printf '%s' "$out"  | grep -oP 'prompt tokens:\s*\K[0-9]+')"
        pf="$(printf '%s' "$out"    | grep -oP 'prefill:.*\(\K[0-9.]+')"
        ttft="$(printf '%s' "$out"  | grep -oP 'prefill:\s*[0-9]+ tokens in \K[0-9.]+')"
        dc="$(printf '%s' "$out"    | grep -oP 'decode:\s*[0-9]+ tokens in [0-9.]+s\s*\(\K[0-9.]+')"
        if [ "$r" -gt 1 ] && [ -n "$pf" ] && [ -n "$dc" ]; then
            pf_sum="$(awk "BEGIN{print $pf_sum + $pf}")"
            dc_sum="$(awk "BEGIN{print $dc_sum + $dc}")"
            ttft_sum="$(awk "BEGIN{print $ttft_sum + $ttft}")"
            n=$((n + 1))
        fi
    done
    if [ "$n" -gt 0 ]; then
        pf_avg="$(awk "BEGIN{printf \"%.0f\", $pf_sum / $n}")"
        dc_avg="$(awk "BEGIN{printf \"%.1f\", $dc_sum / $n}")"
        ttft_avg="$(awk "BEGIN{printf \"%.2f\", $ttft_sum / $n}")"
        printf "%-12s %-16s %-16s %-12s\n" "$ptok" "$pf_avg" "$dc_avg" "$ttft_avg"
    fi
done
