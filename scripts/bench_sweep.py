import argparse
import csv
import json
import logging
import re
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
    "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}

# Mirrors engine::KVCache. Compile time property of the engine.
KV_ELEM_DTYPE = "F8_E4M3"
KV_SCALE_DTYPE = "F32"

# Mirrors model_type_registry() in src/model/config.cpp.
SUPPORTED_MODEL_TYPES = ("llama", "qwen3")

PARA = (
    "The history of computing spans many centuries and involves countless inventors, "
    "mathematicians, and engineers who each contributed pieces to the puzzle of automated "
    "calculation and information processing across generations. "
)

# Lands near llama-bench's 32/256/512/1024/2048 grid plus two deeper points.
LADDER = (1, 8, 15, 30, 60, 140, 300)

# Context cap triggers early rather than late.
CHARS_PER_TOKEN = 3.5

RE_PROMPT = re.compile(r"prompt tokens:\s*(\d+)")
RE_PREFILL = re.compile(r"prefill:\s*\d+ tokens in ([\d.]+)s?\s*\(([\d.]+)")
RE_DECODE = re.compile(r"decode:\s*\d+ tokens in [\d.]+s\s*\(([\d.]+)")


@dataclass(frozen=True)
class Device:
    name: str
    compute_cap: str
    vram_mib: int
    memclk_khz: int
    bus_bits: int
    peak_gbs: float


@dataclass(frozen=True)
class Model:
    path: Path
    model_type: str
    quant: str
    max_context: int
    weight_bytes: int  # VRAM budget
    stream_bytes: int  # traffic per decode step
    kv_bytes: int


@dataclass(frozen=True)
class Row:
    prompt: int
    depth: float
    prefill: tuple[float, float, float]
    prefill_sd: float
    ttft: float
    decode: tuple[float, float, float]
    decode_sd: float
    tpot: float
    gbs: float
    mbu: float


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Median, q1, q3. Ranks are symmetric about the median so q3 never
    collapses onto the maximum at small n. statistics.quantiles interpolates
    and would not reproduce the published numbers."""
    v = sorted(values)
    n = len(v)
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0
    return med, v[int(n * 0.25)], v[n - int(n * 0.25) - 1]


def stdev(values: list[float]) -> float:
    """Sample standard deviation, matching what llama-bench reports so the two
    can share an error bar. The table still shows the IQR, which survives a cold
    first run that this does not."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def query_device(devinfo: Path) -> Device:
    out = subprocess.run([devinfo], capture_output=True, text=True, check=True).stdout
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return Device(
        name=kv["name"],
        compute_cap=kv["compute_cap"],
        vram_mib=int(kv["vram_mib"]),
        memclk_khz=int(kv["memclk_khz"]),
        bus_bits=int(kv["bus_bits"]),
        peak_gbs=float(kv["peak_gbs"]),
    )


def safetensors_bytes(model_dir: Path) -> int:
    """Sum of every tensor on disk. embed_tokens is excluded, decode gathers one
    row from it rather than reading it."""
    total = 0
    for shard in sorted(model_dir.glob("*.safetensors")):
        with shard.open("rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, meta in header.items():
            if name == "__metadata__" or "embed_tokens" in name:
                continue
            elems = 1
            for dim in meta["shape"]:
                elems *= dim
            total += elems * DTYPE_BYTES.get(meta["dtype"], 0)
    return total


def tied_head_bytes(model_dir: Path, tied_embeddings: bool) -> int:
    """embed_tokens size when that tensor doubles as the output projection. A tied
    checkpoint shipping no lm_head reads it in full every decode step, so it counts
    toward traffic even though safetensors_bytes drops it. Zero otherwise.

    Traffic only. The VRAM budget keeps using weight_bytes, whose headroom factor is
    calibrated against that figure."""
    embed_bytes = 0
    has_head = False
    for shard in sorted(model_dir.glob("*.safetensors")):
        with shard.open("rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            has_head |= "lm_head" in name
            if "embed_tokens" not in name:
                continue
            elems = 1
            for dim in meta["shape"]:
                elems *= dim
            embed_bytes += elems * DTYPE_BYTES.get(meta["dtype"], 0)
    return embed_bytes if tied_embeddings and not has_head else 0


def load_model(model_dir: Path) -> Model:
    cfg = json.loads((model_dir / "config.json").read_text())
    model_type = cfg["model_type"]
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise SystemExit(f"{model_dir}: model_type {model_type!r} is not supported")

    layers = cfg["num_hidden_layers"]
    kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    kv_bytes = (
        layers * kv_heads * head_dim * 2 * DTYPE_BYTES[KV_ELEM_DTYPE]
        + layers * kv_heads * 2 * DTYPE_BYTES[KV_SCALE_DTYPE]
    )
    quant = cfg.get("quantization_config", {}).get("quant_method", cfg.get("torch_dtype", "?"))
    weight_bytes = safetensors_bytes(model_dir)
    return Model(
        path=model_dir,
        model_type=model_type,
        quant=quant,
        max_context=cfg["max_position_embeddings"],
        weight_bytes=weight_bytes,
        stream_bytes=weight_bytes + tied_head_bytes(model_dir, cfg.get("tie_word_embeddings", False)),
        kv_bytes=kv_bytes,
    )


def discover_model(models_dir: Path) -> Model:
    for candidate in sorted(models_dir.iterdir()):
        if not (candidate / "config.json").is_file():
            continue
        if not any(candidate.glob("*.safetensors")):
            continue
        try:
            return load_model(candidate)
        except SystemExit:
            continue
    raise SystemExit(f"no supported model under {models_dir}")


def build_ladder(model: Model, device: Device, gen_len: int, ladder: list[int]) -> list[int]:
    """Repeat counts trimmed to what the model and the card can hold. The token
    estimate is deliberately conservative, the binary reports the real count."""
    budget = device.vram_mib * 1024 * 1024 * 0.9 - model.weight_bytes
    kept = []
    for repeats in ladder:
        tokens = len(PARA) * repeats / CHARS_PER_TOKEN + gen_len
        if tokens > model.max_context:
            break
        if budget > 0 and tokens * model.kv_bytes > budget:
            break
        kept.append(repeats)
    if not kept:
        raise SystemExit(f"{model.path.name} does not fit in {device.vram_mib} MiB")
    if len(kept) < len(ladder):
        logger.warning("ladder trimmed to %d points by context or VRAM", len(kept))
    return kept


def run_once(binary: Path, model: Model, prompt_file: Path, args) -> tuple[int, float, float, float] | None:
    proc = subprocess.run(
        [
            binary, model.path,
            "--temperature", str(args.temperature),
            "--max-tokens", str(args.gen_len),
            "--warmup", str(args.engine_warmup),
            "--prompt-file", prompt_file,
            *(["--enforce-eager"] if args.eager else []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        logger.warning("run failed: %s", proc.stderr.strip().splitlines()[-1:] or "no stderr")
        return None

    prompt = RE_PROMPT.search(proc.stdout)
    prefill = RE_PREFILL.search(proc.stdout)
    decode = RE_DECODE.search(proc.stdout)
    if not (prompt and prefill and decode):
        logger.warning("parsed nothing, output format changed?")
        return None
    return (
        int(prompt.group(1)),
        float(prefill.group(2)),
        float(prefill.group(1)),
        float(decode.group(1)),
    )


def measure(binary: Path, model: Model, device: Device, repeats: int, args) -> Row | None:
    prompt_file = Path(args.prompt_file or "").resolve() if args.prompt_file else None
    if prompt_file is None:
        prompt_file = Path(f".bench_prompt_{repeats}.txt")
        prompt_file.write_text(PARA * repeats)

    tokens, prefills, ttfts, decodes = 0, [], [], []
    try:
        for i in range(args.warmup + args.runs):
            result = run_once(binary, model, prompt_file, args)
            if result is None:
                continue
            if i < args.warmup:
                continue
            tokens, pf, ttft, dc = result
            prefills.append(pf)
            ttfts.append(ttft)
            decodes.append(dc)
            if args.run_cooldown:
                time.sleep(args.run_cooldown)
    finally:
        prompt_file.unlink(missing_ok=True)

    if not prefills:
        return None

    ttft_med = quartiles(ttfts)[0]
    decode_stats = quartiles(decodes)
    dc_med = decode_stats[0]

    # Mean cache depth over the generation, the prompt plus half the output.
    depth = tokens + args.gen_len / 2.0
    moved = model.stream_bytes + depth * model.kv_bytes
    return Row(
        prompt=tokens,
        depth=depth,
        prefill=quartiles(prefills),
        prefill_sd=stdev(prefills),
        ttft=ttft_med,
        decode=decode_stats,
        decode_sd=stdev(decodes),
        tpot=1000.0 / dc_med if dc_med else 0.0,
        gbs=moved * dc_med / 1e9,
        mbu=moved * dc_med / (device.peak_gbs * 1e9) * 100.0,
    )


def write_banner(device: Device, model: Model, args) -> None:
    w = sys.stdout.write
    w(f"device 0: {device.name}, compute capability {device.compute_cap}, "
      f"VRAM: {device.vram_mib} MiB\n")
    w(f"bandwidth: {device.peak_gbs} GB/s peak "
      f"({device.bus_bits} bit bus at {device.memclk_khz // 1000} MHz, DDR)\n")
    mode = "eager" if args.eager else "CUDA graph"
    sampler = "argmax" if args.temperature == 0 else f"temperature {args.temperature}"
    w(f"model:    {model.path.name}, {model.quant}, {KV_ELEM_DTYPE} KV, {mode}, {sampler}\n")
    w(f"traffic:  {model.stream_bytes / 1e9:.3f} GB weights per token, "
      f"{model.kv_bytes} B KV per cached token\n\n")
    w(f"{'prompt':<7} {'prefill tok/s':<21} {'TTFT s':<8} {'decode tok/s':<21} "
      f"{'TPOT ms':<8} {'GB/s':<7} {'MBU':<6}\n")
    w("-" * 86 + "\n")


def write_row(row: Row) -> None:
    pf = f"{row.prefill[0]:.0f} [{row.prefill[1]:.0f}-{row.prefill[2]:.0f}]"
    dc = f"{row.decode[0]:.2f} [{row.decode[1]:.2f}-{row.decode[2]:.2f}]"
    sys.stdout.write(
        f"{row.prompt:<7} {pf:<21} {row.ttft:<8.3f} {dc:<21} "
        f"{row.tpot:<8.1f} {row.gbs:<7.1f} {f'{row.mbu:.1f}%':<6}\n"
    )
    sys.stdout.flush()


def write_csv(path: Path, rows: list[Row], device: Device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow([
            "prompt", "depth", "prefill_med", "prefill_q1", "prefill_q3", "prefill_sd",
            "ttft_s", "decode_med", "decode_q1", "decode_q3", "decode_sd",
            "tpot_ms", "gbs", "mbu_pct", "peak_gbs",
        ])
        for r in rows:
            out.writerow([
                r.prompt, f"{r.depth:.1f}", *(f"{v:.4f}" for v in r.prefill),
                f"{r.prefill_sd:.4f}", f"{r.ttft:.4f}",
                *(f"{v:.4f}" for v in r.decode), f"{r.decode_sd:.4f}",
                f"{r.tpot:.1f}", f"{r.gbs:.1f}", f"{r.mbu:.1f}", device.peak_gbs,
            ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, help="model dir, otherwise the first supported one found")
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--binary", type=Path, default=Path("build/runtherder"))
    p.add_argument("--devinfo", type=Path, default=Path("build/runtherder_devinfo"))
    p.add_argument("--runs", type=int, default=7, help="measured samples per row")
    p.add_argument("--warmup", type=int, default=1, help="discarded processes, page cache only")
    p.add_argument("--engine-warmup", type=int, default=3,
                   help="discarded generate() passes per process, pays the cuBLASLt "
                        "heuristic search that every new prompt length triggers")
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--repeats", type=int, nargs="+", default=list(LADDER),
                   help="paragraph repeat counts, one sweep row each")
    p.add_argument("--cooldown", type=float, default=3.0, help="seconds between prompt lengths")
    p.add_argument("--run-cooldown", type=float, default=0.0, help="seconds between runs")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--eager", action="store_true", help="disable CUDA graph capture")
    p.add_argument("--prompt-file", type=Path, help="fixed prompt, disables the ladder sweep")
    p.add_argument("--csv", type=Path, default=Path("bench/latest.csv"),
                   help="raw numbers for plot_bench.py. bench/ is gitignored, "
                        "copy a good run into docs/bench/ to publish it")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="bench_sweep: %(message)s")

    for path in (args.binary, args.devinfo):
        if not path.is_file():
            raise SystemExit(f"{path} not found, build first")

    device = query_device(args.devinfo)
    model = load_model(args.model) if args.model else discover_model(args.models_dir)
    ladder = ([1] if args.prompt_file
              else build_ladder(model, device, args.gen_len, args.repeats))

    write_banner(device, model, args)

    rows, failed = [], 0
    for repeats in ladder:
        row = measure(args.binary, model, device, repeats, args)
        if row is None:
            failed += 1
            sys.stdout.write(f"{'?':<7} {'FAILED':<21} {'-':<8} {'-':<21} "
                             f"{'-':<8} {'-':<7} {'-':<6}\n")
            sys.stdout.flush()
        else:
            rows.append(row)
            write_row(row)
        time.sleep(args.cooldown)

    if args.csv and rows:
        write_csv(args.csv, rows, device)
        logger.info("wrote %s", args.csv)
    if failed:
        logger.warning("%d runs failed or did not parse, rows above are degraded", failed)


if __name__ == "__main__":
    main()
