"""
Generic benchmarking harness for edge-ai-benchmarks.

Usage examples:
    python scripts/harness.py --model mobilenetv2_fp32 --image models/test1.jpg
    python scripts/harness.py --model mobilenetv2_fp32 --backend xnnpack --threads 4
    python scripts/harness.py --model mobilenetv2_int8 --backend vx_npu --board advantech-rsb3720

Run from the repo root (the directory containing scripts/, models/, results/).

Design: this file contains everything that is the SAME for every model and
every board — loading the interpreter, choosing a backend/delegate, the
warmup + timed inference loop, latency/memory measurement, and CSV logging.
Anything model-specific (preprocessing, output decoding) lives in
scripts/adapters/<model_name>.py and is loaded dynamically via --model.
"""

import argparse
import csv
import importlib
import os
import resource
import socket
import sys
import time
from datetime import datetime, timezone

import ai_edge_litert.interpreter as tflite

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Needed so "scripts.adapters.<model>" resolves as a package regardless of
# whether you ran this as `python scripts/harness.py` (cwd-independent,
# per the __file__ discussion) or from elsewhere.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
RESULTS_CSV = os.path.join(REPO_ROOT, "results", "benchmarks.csv")
CSV_FIELDS = [
    "timestamp_utc", "board", "model", "backend", "threads",
    "median_ms", "p95_ms", "peak_rss_mb", "top1_label", "top1_score",
]

# Backend -> delegate loader. "cpu" and "xnnpack" need no external .so;
# XNNPACK is LiteRT's built-in CPU accelerator, enabled via num_threads
# and its own default delegate, not a separate load_delegate() call.
# vx_npu is board-specific (NXP eIQ VX delegate on the Advantech i.MX 8M Plus)
# and needs the actual .so path on that board — adjust VX_DELEGATE_PATH
# below to match your BSP if it differs.
VX_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"


def load_adapter(model_name):
    module = importlib.import_module(f"scripts.adapters.{model_name}")
    return module


def build_interpreter(model_path, backend, threads):
    delegates = []
    if backend == "vx_npu":
        if not os.path.exists(VX_DELEGATE_PATH):
            sys.exit(
                f"vx_npu backend requested but delegate not found at "
                f"{VX_DELEGATE_PATH}. Check your BSP's delegate path and "
                f"update VX_DELEGATE_PATH in harness.py."
            )
        delegates.append(tflite.load_delegate(VX_DELEGATE_PATH))
    elif backend not in ("cpu", "xnnpack"):
        sys.exit(f"Unknown backend '{backend}'. Use cpu, xnnpack, or vx_npu.")

    kwargs = {"model_path": model_path}
    if delegates:
        kwargs["experimental_delegates"] = delegates
    if backend in ("cpu", "xnnpack") and threads:
        kwargs["num_threads"] = threads

    interp = tflite.Interpreter(**kwargs)
    interp.allocate_tensors()
    return interp


def run_benchmark(args):
    adapter = load_adapter(args.model)

    # Fail fast rather than silently benchmarking garbage: NPU delegates
    # generally require an int8-quantized model.
    if args.backend == "vx_npu" and getattr(adapter, "IS_QUANTIZED", None) is False:
        sys.exit(
            f"Model '{args.model}' is not quantized (IS_QUANTIZED=False) but "
            f"--backend vx_npu was requested. NPU delegates typically require "
            f"an int8 model. Convert/quantize first, or use --backend cpu/xnnpack."
        )

    model_path = os.path.join(REPO_ROOT, adapter.MODEL_PATH)
    image_path = os.path.join(REPO_ROOT, args.image) if not os.path.isabs(args.image) else args.image

    interp = build_interpreter(model_path, args.backend, args.threads)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    x = adapter.preprocess(image_path)

    def run_once():
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"])[0]

    for _ in range(args.warmup):
        run_once()

    times = []
    last_output = None
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        last_output = run_once()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times.sort()
    n = len(times)
    median = times[n // 2]
    p95 = times[int(n * 0.95)]
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    top5 = adapter.decode(last_output, top=5)

    print(f"Model:    {args.model}")
    print(f"Backend:  {args.backend} (threads={args.threads})")
    print(f"Image:    {os.path.basename(image_path)}")
    print("Top-5:")
    for label, score in top5:
        print(f"  {label:30s} {score:.4f}")
    print(f"median={median:.2f} ms  p95={p95:.2f} ms  peak_rss={peak_rss_mb:.1f} MB")

    log_result(args, median, p95, peak_rss_mb, top5[0])


def log_result(args, median, p95, peak_rss_mb, top1):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "board": args.board,
            "model": args.model,
            "backend": args.backend,
            "threads": args.threads,
            "median_ms": f"{median:.2f}",
            "p95_ms": f"{p95:.2f}",
            "peak_rss_mb": f"{peak_rss_mb:.1f}",
            "top1_label": top1[0],
            "top1_score": f"{top1[1]:.4f}",
        })
    print(f"Logged to {RESULTS_CSV}")


def parse_args():
    p = argparse.ArgumentParser(description="Generic edge-ai-benchmarks harness")
    p.add_argument("--model", required=True, help="adapter name, e.g. mobilenetv2_fp32 (must exist in scripts/adapters/)")
    p.add_argument("--image", default=os.path.join("models", "test1.jpg"), help="path to test image, relative to repo root")
    p.add_argument("--backend", default="cpu", choices=["cpu", "xnnpack", "vx_npu"], help="cpu/xnnpack for Pi 4, vx_npu for Advantech")
    p.add_argument("--threads", type=int, default=1, help="num_threads for cpu/xnnpack backends")
    p.add_argument("--warmup", type=int, default=100, help="number of untimed warmup runs")
    p.add_argument("--iterations", type=int, default=500, help="number of timed runs")
    p.add_argument("--board", default=socket.gethostname(), help="board label for the CSV row (defaults to hostname)")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
