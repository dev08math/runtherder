#!/usr/bin/env bash
# Throughput sweep shaped to line up with llama-bench rows.
#
# pp<N>        prompt processing, N prompt tokens, t/s = N / TTFT
# tg<N> @d<D>  token generation, N tokens decoded on top of D cached tokens
#
# llama-bench tg runs from an EMPTY cache unless -d is given, so tg128 there
# compares to tg128 @d0 here. Match the depth before comparing.
#
# llama-bench also runs NO sampler, test_gen takes rand() % n_vocab as the next
# token. Our tg carries a real argmax over the vocab every step, so our t/s
# covers strictly more work.
#
# Env overrides: RUNS, WARMUP, ENGINE_WARMUP, COOLDOWN, GEN_LEN, REPEATS, TEMPERATURE, EAGER.
# Bandwidth is NOT an override. It comes from runtherder_devinfo off the driver.
#   RUNS=3 ./scripts/bench_sweep.sh           quick pass
#   RUNS=15 ./scripts/bench_sweep.sh          publication pass
set -uo pipefail
export LC_ALL=C

MODEL="${1:-./models/Llama-3.2-3B-Instruct-quantized.w8a8}"
BIN=./build/runtherder
DEVINFO=./build/runtherder_devinfo
RUNS="${RUNS:-7}"
WARMUP="${WARMUP:-1}"
COOLDOWN="${COOLDOWN:-3}"
ENGINE_WARMUP="${ENGINE_WARMUP:-3}"
GEN_LEN="${GEN_LEN:-128}"

TEMPERATURE="${TEMPERATURE:-0.0}"
EAGER="${EAGER:-}"

PARA="The history of computing spans many centuries and involves countless inventors, mathematicians, and engineers who each contributed pieces to the puzzle of automated calculation and information processing across generations. "
read -r -a REPEATS <<< "${REPEATS:-1 8 15 30 60 140 300}"

PROMPT_FILE="$(mktemp)"
ERR_FILE="$(mktemp)"
trap 'rm -f "$PROMPT_FILE" "$ERR_FILE"' EXIT

if [ ! -x "$BIN" ]; then
    printf 'bench_sweep: %s not found, build first\n' "$BIN" >&2
    exit 1
fi
if [ ! -x "$DEVINFO" ]; then
    printf 'bench_sweep: %s not found, build first\n' "$DEVINFO" >&2
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    printf 'bench_sweep: model dir %s not found\n' "$MODEL" >&2
    exit 1
fi

# Median with the interquartile range, $1 decimal places. Quartile ranks are
# symmetric about the median so q3 never collapses onto the maximum at small n.
stats() {
    sort -g | awk -v d="$1" '
        { v[NR] = $1 + 0 }
        END {
            if (NR == 0) { printf "n/a"; exit }
            med = (NR % 2) ? v[int(NR / 2) + 1] : (v[NR / 2] + v[NR / 2 + 1]) / 2.0
            q1  = int(NR * 0.25) + 1
            q3  = NR - int(NR * 0.25)
            printf "%.*f [%.*f-%.*f]", d, med, d, v[q1], d, v[q3]
        }'
}

median_only() {
    sort -g | awk '
        { v[NR] = $1 + 0 }
        END {
            if (NR == 0) { print 0; exit }
            print (NR % 2) ? v[int(NR / 2) + 1] : (v[NR / 2] + v[NR / 2 + 1]) / 2.0
        }'
}

read -r W_BYTES KV_BYTES KV_LABEL <<< "$(python3 - "$MODEL" <<'PY'
import json, pathlib, struct, sys

SIZES = {"F64":8,"F32":4,"F16":2,"BF16":2,"I64":8,"I32":4,"I16":2,"I8":1,
         "U8":1,"BOOL":1,"F8_E4M3":1,"F8_E5M2":1}

# Mirrors engine::KVCache: E4M3 k and v slabs plus one fp32 dequant scale each
# per (token, kv_head). Update both together.
KV_ELEM_DTYPE  = "F8_E4M3"
KV_SCALE_DTYPE = "F32"
d = pathlib.Path(sys.argv[1])
weights = 0
for f in sorted(d.glob("*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    for name, meta in head.items():
        if name == "__metadata__" or "embed_tokens" in name:
            continue
        nelem = 1
        for s in meta["shape"]:
            nelem *= s
        weights += nelem * SIZES.get(meta["dtype"], 0)

cfg = json.loads((d / "config.json").read_text())
layers = cfg["num_hidden_layers"]
kvh = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
kv = (layers * kvh * hd * 2 * SIZES[KV_ELEM_DTYPE]
      + layers * kvh * 2 * SIZES[KV_SCALE_DTYPE])
print(weights, kv, KV_ELEM_DTYPE)
PY
)"
W_BYTES="${W_BYTES:-0}"
KV_BYTES="${KV_BYTES:-0}"

DEV_NAME=""; DEV_CC=""; DEV_VRAM=""; PEAK_GBS=""
BUS_BITS=""; MEMCLK_KHZ=""
while IFS='=' read -r key val; do
    case "$key" in
        name)         DEV_NAME="$val" ;;
        compute_cap)  DEV_CC="$val" ;;
        vram_mib)     DEV_VRAM="$val" ;;
        memclk_khz)   MEMCLK_KHZ="$val" ;;
        bus_bits)     BUS_BITS="$val" ;;
        peak_gbs)     PEAK_GBS="$val" ;;
    esac
done < <("$DEVINFO")

if [ -z "$PEAK_GBS" ]; then
    printf 'bench_sweep: %s reported no peak_gbs, cannot compute MBU\n' "$DEVINFO" >&2
    exit 1
fi

if [ -n "$EAGER" ]; then EXEC_MODE="eager"; else EXEC_MODE="CUDA graph"; fi
if awk "BEGIN{exit !($TEMPERATURE == 0)}"; then
    SAMPLER="argmax"
else
    SAMPLER="temperature $TEMPERATURE"
fi

printf 'device 0: %s, compute capability %s, VRAM: %s MiB\n' \
    "$DEV_NAME" "$DEV_CC" "$DEV_VRAM"
printf 'bandwidth: %s GB/s peak (%d bit bus at %d MHz, DDR)\n' \
    "$PEAK_GBS" "$BUS_BITS" "$((MEMCLK_KHZ / 1000))"
printf 'model:    %s, %s KV, %s, %s\n' \
    "$(basename "$MODEL")" "$KV_LABEL" "$EXEC_MODE" "$SAMPLER"
printf 'traffic:  %.3f GB weights per token, %d B KV per cached token\n\n' \
    "$(awk "BEGIN{print $W_BYTES / 1e9}")" "$KV_BYTES"

printf '%-7s %-21s %-8s %-21s %-8s %-6s\n' \
    "prompt" "prefill tok/s" "TTFT s" "decode tok/s" "TPOT ms" "MBU"
printf -- '--------------------------------------------------------------------------------------\n'

total_fail=0
for rep in "${REPEATS[@]}"; do
    python3 -c "import sys; sys.stdout.write('${PARA}' * ${rep})" > "$PROMPT_FILE"

    pf_vals=(); dc_vals=(); tt_vals=(); ptok=""; fails=0

    for r in $(seq 1 $((WARMUP + RUNS))); do
        if ! out="$("$BIN" "$MODEL" --temperature "$TEMPERATURE" --max-tokens "$GEN_LEN" \
                    --warmup "$ENGINE_WARMUP" --prompt-file "$PROMPT_FILE" \
                    ${EAGER:+"$EAGER"} 2> "$ERR_FILE")"; then
            fails=$((fails + 1))
            printf 'bench_sweep: run %d failed (rep %s): %s\n' \
                "$r" "$rep" "$(tail -n 1 "$ERR_FILE")" >&2
            continue
        fi
        if [ "$r" -le "$WARMUP" ]; then
            continue
        fi

        ptok="$(printf '%s' "$out" | grep -oP 'prompt tokens:\s*\K[0-9]+')"
        pf="$(printf '%s' "$out"   | grep -oP 'prefill:.*\(\K[0-9.]+')"
        ttft="$(printf '%s' "$out" | grep -oP 'prefill:\s*[0-9]+ tokens in \K[0-9.]+')"
        dc="$(printf '%s' "$out"   | grep -oP 'decode:\s*[0-9]+ tokens in [0-9.]+s\s*\(\K[0-9.]+')"

        if [ -z "$pf" ] || [ -z "$dc" ] || [ -z "$ttft" ]; then
            fails=$((fails + 1))
            printf 'bench_sweep: run %d parsed nothing, output format changed?\n' "$r" >&2
            continue
        fi
        pf_vals+=("$pf"); dc_vals+=("$dc"); tt_vals+=("$ttft")
    done

    total_fail=$((total_fail + fails))
    n="${#pf_vals[@]}"
    if [ "$n" -eq 0 ]; then
        printf '%-7s %-21s %-8s %-21s %-8s %-6s\n' \
            "${ptok:-?}" "FAILED" "-" "-" "-" "-"
        sleep "$COOLDOWN"
        continue
    fi

    ttft_med="$(printf '%s\n' "${tt_vals[@]}" | median_only)"
    dc_med="$(printf '%s\n' "${dc_vals[@]}" | median_only)"

    # Mean cache depth over the generation, the prompt plus half the output.
    mbu="$(awk "BEGIN{
        depth = $ptok + $GEN_LEN / 2.0
        bytes = $W_BYTES + depth * $KV_BYTES
        printf \"%.1f%%\", bytes * $dc_med / ($PEAK_GBS * 1e9) * 100.0
    }")"
    printf '%-7s %-21s %-8s %-21s %-8s %-6s\n' \
        "$ptok" \
        "$(printf '%s\n' "${pf_vals[@]}" | stats 0)" \
        "$(awk "BEGIN{printf \"%.3f\", $ttft_med}")" \
        "$(printf '%s\n' "${dc_vals[@]}" | stats 2)" \
        "$(awk "BEGIN{printf \"%.1f\", ($dc_med > 0) ? 1000.0 / $dc_med : 0}")" \
        "$mbu"

    sleep "$COOLDOWN"
done

if [ "$total_fail" -gt 0 ]; then
    printf 'WARNING: %d runs failed or did not parse, rows above are degraded\n' "$total_fail" >&2
fi
